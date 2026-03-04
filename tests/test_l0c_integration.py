"""
L0C 集成测试

验证 MAC 在 L0C doublebuffer 满载时必须等待 FIXPIPE 释放槽位。
覆盖 DEFAULT 路径 (matmulFull / matmulN / matmulK) 及 PRELOAD_1/PRELOAD_2 路径。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline


# ────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────────────────────

def mac_p_events(timeline):
    return sorted([e for e in timeline if e.unit == "MAC" and e.operation == "P"],
                  key=lambda e: e.start_time)

def mac_o_events(timeline):
    return sorted([e for e in timeline if e.unit == "MAC" and e.operation == "O"],
                  key=lambda e: e.start_time)

def fix_p_events(timeline):
    return sorted([e for e in timeline if e.unit == "FIXPIPE" and e.operation == "P"],
                  key=lambda e: e.start_time)

def fix_o_events(timeline):
    return sorted([e for e in timeline if e.unit == "FIXPIPE" and e.operation == "O"],
                  key=lambda e: e.start_time)


def make_default(s2_total=512, s2_base=256, **kw):
    """DEFAULT 路径：注意 s2_base > baseN_C1(128) 会触发 matmulN"""
    return C1Modeler(
        s1_total=128, s2_total=s2_total, d_total=128,
        s1_base_size=128, s2_base_size=s2_base, d_base_size=128,
        baseM_C1=128, baseN_C1=128, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        is_l2cache=True,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
        L1_db=False, L0_db=False,
        **kw,
    )


def make_matmulfull(k_blocks=2):
    """强制 matmulFull：s2_base==baseN_C1, d_base==baseK_C1"""
    return C1Modeler(
        s1_total=128, s2_total=128 * k_blocks, d_total=128,
        s1_base_size=128, s2_base_size=128, d_base_size=128,
        baseM_C1=128, baseN_C1=128, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        is_l2cache=True,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
        L1_db=False, L0_db=False,
    )


# ────────────────────────────────────────────────────────────────────────────
# 基础完整性测试：每个 MAC 事件之后对应一个 FIXPIPE
# ────────────────────────────────────────────────────────────────────────────

class TestL0CBasicOrdering:
    """时间线基本时序约束"""

    def test_fixpipe_p_after_mac_p(self):
        """FIXPIPE-P 必须在对应 MAC-P 结束后开始（matmulFull：1:1 对应关系）"""
        m = make_matmulfull(k_blocks=2)
        timeline, _, _, _ = m.run_simulation()
        macs = mac_p_events(timeline)
        fixs = fix_p_events(timeline)
        assert len(macs) == len(fixs) > 0
        for i, (mac, fix) in enumerate(zip(macs, fixs)):
            assert fix.start_time >= mac.end_time, (
                f"FIXPIPE-P[{i}] starts at {fix.start_time:.1f} before MAC-P[{i}] ends at {mac.end_time:.1f}"
            )

    def test_fixpipe_o_after_mac_o(self):
        """FIXPIPE-O 必须在对应 MAC-O 结束后开始（matmulFull：1:1 对应关系）"""
        m = make_matmulfull(k_blocks=2)
        timeline, _, _, _ = m.run_simulation()
        macs = mac_o_events(timeline)
        fixs = fix_o_events(timeline)
        assert len(macs) == len(fixs) > 0
        for i, (mac, fix) in enumerate(zip(macs, fixs)):
            assert fix.start_time >= mac.end_time, (
                f"FIXPIPE-O[{i}] starts at {fix.start_time:.1f} before MAC-O[{i}] ends at {mac.end_time:.1f}"
            )

    def test_mac_events_non_overlapping(self):
        """MAC 事件（共用 MAC 单元）不得重叠"""
        m = make_default()
        timeline, _, _, _ = m.run_simulation()
        all_macs = sorted([e for e in timeline if e.unit == "MAC"], key=lambda e: e.start_time)
        for i in range(len(all_macs) - 1):
            assert all_macs[i].end_time <= all_macs[i + 1].start_time + 1e-9, (
                f"MAC events overlap: [{i}]({all_macs[i].start_time:.1f}-{all_macs[i].end_time:.1f}) "
                f"vs [{i+1}]({all_macs[i+1].start_time:.1f}-{all_macs[i+1].end_time:.1f})"
            )

    def test_fixpipe_events_non_overlapping(self):
        """FIXPIPE 事件（共用 FIXPIPE 单元）不得重叠"""
        m = make_default()
        timeline, _, _, _ = m.run_simulation()
        all_fix = sorted([e for e in timeline if e.unit == "FIXPIPE"], key=lambda e: e.start_time)
        for i in range(len(all_fix) - 1):
            assert all_fix[i].end_time <= all_fix[i + 1].start_time + 1e-9, (
                f"FIXPIPE events overlap: [{i}]({all_fix[i].start_time:.1f}-{all_fix[i].end_time:.1f}) "
                f"vs [{i+1}]({all_fix[i+1].start_time:.1f}-{all_fix[i+1].end_time:.1f})"
            )


# ────────────────────────────────────────────────────────────────────────────
# matmulFull 路径：每对 (MAC-P, FIXPIPE-P) 对应一次完整 L0C 分配/释放
# ────────────────────────────────────────────────────────────────────────────

class TestL0CMatmulFull:
    """DEFAULT 路径 matmulFull 的 L0C 集成"""

    def test_matmul_full_mac_fixpipe_pairs(self):
        """matmulFull：MAC-P 个数 == FIXPIPE-P 个数 == k_block_count"""
        k_blocks = 4
        m = make_default(s2_total=128 * k_blocks, s2_base=128)
        timeline, _, _, _ = m.run_simulation()
        macs = mac_p_events(timeline)
        fixs = fix_p_events(timeline)
        assert len(macs) == k_blocks
        assert len(fixs) == k_blocks

    def test_matmul_full_c2_mac_fixpipe_pairs(self):
        """matmulFull C2：MAC-O 个数 == FIXPIPE-O 个数 == k_block_count"""
        k_blocks = 3
        m = make_default(s2_total=128 * k_blocks, s2_base=128)
        timeline, _, _, _ = m.run_simulation()
        macs = mac_o_events(timeline)
        fixs = fix_o_events(timeline)
        assert len(macs) == k_blocks
        assert len(fixs) == k_blocks

    def test_second_mac_p_waits_if_doublebuffer_full(self):
        """当 k_blocks >= 3 时，第 3 个 MAC-P 不能早于第 1 个 FIXPIPE-P"""
        # doublebuffer 2 slots：第 3 个 MAC 必须等第 1 个 FIXPIPE 释放
        k_blocks = 4
        m = make_default(s2_total=128 * k_blocks, s2_base=128)
        timeline, _, _, _ = m.run_simulation()
        macs = mac_p_events(timeline)
        fixs = fix_p_events(timeline)

        if len(macs) >= 3 and len(fixs) >= 1:
            # 第 3 个 MAC-P 的开始时间 >= 第 1 个 FIXPIPE-P 的结束时间
            assert macs[2].start_time >= fixs[0].end_time, (
                f"MAC-P[2] (start={macs[2].start_time:.1f}) should wait for "
                f"FIXPIPE-P[0] (end={fixs[0].end_time:.1f})"
            )


# ────────────────────────────────────────────────────────────────────────────
# matmulK 路径：同一 k_block 内多个 sub-MAC 共用一个 L0C 槽，一次 FIXPIPE
# ────────────────────────────────────────────────────────────────────────────

class TestL0CMatmulK:
    """matmulK 路径的 L0C 集成（d_base > baseK_C1 触发）"""

    def make_matmulk(self):
        """d_base(256) > baseK_C1(128) → matmulK：每块 2 个 sub-MAC，1 个 FIXPIPE"""
        return C1Modeler(
            s1_total=128, s2_total=256, d_total=256,
            s1_base_size=128, s2_base_size=256, d_base_size=256,
            baseM_C1=128, baseN_C1=256, baseK_C1=128,
            baseM_C2=128, baseN_C2=128, baseK_C2=128,
            q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
            is_l2cache=True,
            inter_core_pipeline=InterCorePipeline.DEFAULT,
            inner_core_pipeline=InnerCorePipeline.DEFAULT,
            L1_db=False, L0_db=False,
        )

    def test_matmulk_more_macs_than_fixpipes(self):
        """matmulK：MAC-P 数 > FIXPIPE-P 数（多 sub-MAC 共用一次 FIXPIPE）"""
        m = self.make_matmulk()
        timeline, _, _, _ = m.run_simulation()
        macs = mac_p_events(timeline)
        fixs = fix_p_events(timeline)
        # d_base=256, baseK=128 → 2 sub-MACs per k_block，1 k_block → 2 MACs, 1 FIXPIPE
        assert len(macs) > len(fixs), (
            f"matmulK: expected more MAC-P ({len(macs)}) than FIXPIPE-P ({len(fixs)})"
        )
        assert len(fixs) >= 1

    def test_matmulk_fixpipe_after_all_submacs(self):
        """matmulK：FIXPIPE-P 在最后一个 sub-MAC 完成后才开始"""
        m = self.make_matmulk()
        timeline, _, _, _ = m.run_simulation()
        macs = mac_p_events(timeline)
        fixs = fix_p_events(timeline)
        if len(macs) >= 2 and len(fixs) >= 1:
            # 第 1 个 FIXPIPE 应晚于第 2 个 MAC（最后的 sub-MAC）
            assert fixs[0].start_time >= macs[1].end_time, (
                f"FIXPIPE-P[0] should start after MAC-P[1] ends: "
                f"fix={fixs[0].start_time:.1f}, mac_end={macs[1].end_time:.1f}"
            )

    def test_matmulk_runs(self):
        """matmulK 路径正常运行"""
        m = self.make_matmulk()
        _, _, _, total = m.run_simulation()
        assert total > 0


# ────────────────────────────────────────────────────────────────────────────
# matmulN 路径：沿 N 轴切分，多 sub-MAC 写不同列，一次 FIXPIPE
# ────────────────────────────────────────────────────────────────────────────

class TestL0CMatmulN:
    """matmulN 路径的 L0C 集成（s2_base > baseN_C1 触发）"""

    def make_matmuln(self):
        """s2_base(512) > baseN_C1(128) → matmulN：4 个 sub-MAC，1 个 FIXPIPE"""
        return C1Modeler(
            s1_total=128, s2_total=512, d_total=128,
            s1_base_size=128, s2_base_size=512, d_base_size=128,
            baseM_C1=128, baseN_C1=128, baseK_C1=128,
            baseM_C2=128, baseN_C2=128, baseK_C2=128,
            q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
            is_l2cache=True,
            inter_core_pipeline=InterCorePipeline.DEFAULT,
            inner_core_pipeline=InnerCorePipeline.DEFAULT,
            L1_db=False, L0_db=False,
        )

    def test_matmuln_more_macs_than_fixpipes(self):
        """matmulN：MAC-P 数 > FIXPIPE-P 数"""
        m = self.make_matmuln()
        timeline, _, _, _ = m.run_simulation()
        macs = mac_p_events(timeline)
        fixs = fix_p_events(timeline)
        assert len(macs) > len(fixs), (
            f"matmulN: expected more MAC-P ({len(macs)}) than FIXPIPE-P ({len(fixs)})"
        )

    def test_matmuln_runs(self):
        """matmulN 路径正常运行"""
        m = self.make_matmuln()
        _, _, _, total = m.run_simulation()
        assert total > 0


# ────────────────────────────────────────────────────────────────────────────
# PRELOAD_1 + L0C 集成
# ────────────────────────────────────────────────────────────────────────────

class TestL0CPreload1:
    """PRELOAD_1 路径的 L0C 集成"""

    def make_preload1(self, k_blocks=4):
        s2 = 128 * k_blocks
        return C1Modeler(
            s1_total=128, s2_total=s2, d_total=128,
            s1_base_size=128, s2_base_size=128, d_base_size=128,
            baseM_C1=128, baseN_C1=128, baseK_C1=128,
            baseM_C2=128, baseN_C2=128, baseK_C2=128,
            q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
            is_l2cache=True,
            inter_core_pipeline=InterCorePipeline.PRELOAD_1,
            inner_core_pipeline=InnerCorePipeline.DEFAULT,
            L1_db=False, L0_db=False,
        )

    def test_preload1_mac_fixpipe_no_overlap(self):
        """PRELOAD_1：MAC 和 FIXPIPE 无重叠"""
        m = self.make_preload1()
        timeline, _, _, _ = m.run_simulation()
        all_macs = sorted([e for e in timeline if e.unit == "MAC"], key=lambda e: e.start_time)
        for i in range(len(all_macs) - 1):
            assert all_macs[i].end_time <= all_macs[i + 1].start_time + 1e-9

    def test_preload1_l0c_doublebuffer_respected(self):
        """PRELOAD_1：4 个 K 块时，第 3 个 MAC-P 不早于第 1 个 FIXPIPE-P"""
        m = self.make_preload1(k_blocks=4)
        timeline, _, _, _ = m.run_simulation()
        macs = mac_p_events(timeline)
        fixs = fix_p_events(timeline)
        if len(macs) >= 3 and len(fixs) >= 1:
            assert macs[2].start_time >= fixs[0].end_time, (
                f"PRELOAD_1 L0C: MAC-P[2] (start={macs[2].start_time:.1f}) should not "
                f"start before FIXPIPE-P[0] ends ({fixs[0].end_time:.1f})"
            )

    def test_preload1_positive_cycles(self):
        """PRELOAD_1 正常运行"""
        m = self.make_preload1()
        _, _, _, total = m.run_simulation()
        assert total > 0


# ────────────────────────────────────────────────────────────────────────────
# PRELOAD_2 + L0C 集成
# ────────────────────────────────────────────────────────────────────────────

class TestL0CPreload2:
    """PRELOAD_2 路径的 L0C 集成"""

    def make_preload2(self, k_blocks=4):
        s2 = 128 * k_blocks
        return C1Modeler(
            s1_total=128, s2_total=s2, d_total=128,
            s1_base_size=128, s2_base_size=128, d_base_size=128,
            baseM_C1=128, baseN_C1=128, baseK_C1=128,
            baseM_C2=128, baseN_C2=128, baseK_C2=128,
            q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
            is_l2cache=True,
            inter_core_pipeline=InterCorePipeline.PRELOAD_2,
            inner_core_pipeline=InnerCorePipeline.DEFAULT,
            L1_db=False, L0_db=False,
        )

    def test_preload2_mac_no_overlap(self):
        """PRELOAD_2：MAC 事件无重叠"""
        m = self.make_preload2()
        timeline, _, _, _ = m.run_simulation()
        all_macs = sorted([e for e in timeline if e.unit == "MAC"], key=lambda e: e.start_time)
        for i in range(len(all_macs) - 1):
            assert all_macs[i].end_time <= all_macs[i + 1].start_time + 1e-9

    def test_preload2_l0c_doublebuffer_respected(self):
        """PRELOAD_2：4 个 K 块时，第 3 个 MAC-P 不早于第 1 个 FIXPIPE-P"""
        m = self.make_preload2(k_blocks=4)
        timeline, _, _, _ = m.run_simulation()
        macs = mac_p_events(timeline)
        fixs = fix_p_events(timeline)
        if len(macs) >= 3 and len(fixs) >= 1:
            assert macs[2].start_time >= fixs[0].end_time, (
                f"PRELOAD_2 L0C: MAC-P[2] (start={macs[2].start_time:.1f}) should not "
                f"start before FIXPIPE-P[0] ends ({fixs[0].end_time:.1f})"
            )

    def test_preload2_fixpipe_no_overlap(self):
        """PRELOAD_2：FIXPIPE 事件无重叠"""
        m = self.make_preload2()
        timeline, _, _, _ = m.run_simulation()
        all_fix = sorted([e for e in timeline if e.unit == "FIXPIPE"], key=lambda e: e.start_time)
        for i in range(len(all_fix) - 1):
            assert all_fix[i].end_time <= all_fix[i + 1].start_time + 1e-9

    def test_preload2_positive_cycles(self):
        """PRELOAD_2 正常运行"""
        m = self.make_preload2()
        _, _, _, total = m.run_simulation()
        assert total > 0

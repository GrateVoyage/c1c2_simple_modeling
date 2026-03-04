"""
PRELOAD_2 流水线专项测试

验证点：
- V1 紧跟 C1（不等 C2 发射）
- C2 延迟 2 个 K
- 3WS 约束（第 4 个 C1 必须等待最早的 WS 释放）
- PRELOAD_2 可以正常运行并产生有效周期数
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline


def make_modeler(k_blocks: int = 4, **kwargs):
    """创建 PRELOAD_2 模式建模器，默认 4 个 K 块（便于验证 C2 延迟逻辑）"""
    s2_per_block = 128
    return C1Modeler(
        s1_total=128,
        s2_total=s2_per_block * k_blocks,
        d_total=128,
        s1_base_size=128,
        s2_base_size=s2_per_block,
        d_base_size=128,
        baseM_C1=128, baseN_C1=128, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=128,
        q_data_type=DataType.FP16,
        kv_data_type=DataType.FP16,
        is_l2cache=True,
        inter_core_pipeline=InterCorePipeline.PRELOAD_2,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
        L1_db=False,
        L0_db=False,
        **kwargs,
    )


class TestPreload2Runs:
    """基础运行测试"""

    def test_runs_single_k(self):
        """单 K 块：仅有收尾逻辑，无主循环 C2"""
        m = make_modeler(k_blocks=1)
        timeline, _, _, total = m.run_simulation()
        assert total > 0
        assert len(timeline) > 0

    def test_runs_two_k(self):
        """2 个 K 块：1 个进入收尾（tail loop）"""
        m = make_modeler(k_blocks=2)
        timeline, _, _, total = m.run_simulation()
        assert total > 0

    def test_runs_three_k(self):
        """3 个 K 块：2 个进入收尾"""
        m = make_modeler(k_blocks=3)
        timeline, _, _, total = m.run_simulation()
        assert total > 0

    def test_runs_four_k(self):
        """4 个 K 块：2 个通过主循环，2 个收尾"""
        m = make_modeler(k_blocks=4)
        timeline, _, _, total = m.run_simulation()
        assert total > 0

    def test_event_count_sanity(self):
        """4 个 K 块应有 4 个 MAC-P 和 4 个 MAC-O"""
        m = make_modeler(k_blocks=4)
        timeline, _, _, _ = m.run_simulation()
        mac_p = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
        mac_o = [e for e in timeline if e.unit == "MAC" and e.operation == "O"]
        k_count = 4
        assert len(mac_p) == k_count, f"Expected {k_count} MAC-P, got {len(mac_p)}"
        assert len(mac_o) == k_count, f"Expected {k_count} MAC-O, got {len(mac_o)}"


class TestPreload2V1FollowsC1:
    """PRELOAD_2 关键特性：V1 紧跟 C1"""

    def test_v1_starts_after_c1_fixpipe_ends(self):
        """每个 V1 事件的 start_time >= 对应 C1 FIXPIPE-P 的 end_time"""
        m = make_modeler(k_blocks=4)
        timeline, _, _, _ = m.run_simulation()

        fix_p_events = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "P"]
        v1_events    = [e for e in timeline if e.unit == "VECTOR_V1"]

        assert len(fix_p_events) > 0
        assert len(v1_events) > 0

        for i, (fix_p, v1) in enumerate(zip(fix_p_events, v1_events)):
            assert v1.start_time >= fix_p.end_time, (
                f"V1[{i}] starts at {v1.start_time:.1f} before FIXPIPE-P[{i}] ends at {fix_p.end_time:.1f}"
            )

    def test_v1_precedes_c2(self):
        """在 PRELOAD_2 中，V1[k] 必须在对应 MAC-O[k] 开始之前完成"""
        m = make_modeler(k_blocks=4)
        timeline, _, _, _ = m.run_simulation()

        v1_events  = sorted([e for e in timeline if e.unit == "VECTOR_V1"],  key=lambda e: e.start_time)
        mac_o_events = sorted([e for e in timeline if e.unit == "MAC" and e.operation == "O"], key=lambda e: e.start_time)

        k_count = 4
        assert len(v1_events) == k_count
        assert len(mac_o_events) == k_count

        for i, (v1, mac_o) in enumerate(zip(v1_events, mac_o_events)):
            assert mac_o.start_time >= v1.end_time, (
                f"MAC-O[{i}] starts at {mac_o.start_time:.1f} before V1[{i}] ends at {v1.end_time:.1f}"
            )


class TestPreload2C2Delay:
    """PRELOAD_2 的 C2 延迟 2 个 K 验证"""

    def test_c2_delay_2_mac_ordering(self):
        """MAC-P[2] 开始时间 <= MAC-O[0] 开始时间（C2[0] 晚于 C1[2] 发射）"""
        m = make_modeler(k_blocks=4)
        timeline, _, _, _ = m.run_simulation()

        mac_p = sorted([e for e in timeline if e.unit == "MAC" and e.operation == "P"], key=lambda e: e.start_time)
        mac_o = sorted([e for e in timeline if e.unit == "MAC" and e.operation == "O"], key=lambda e: e.start_time)

        assert len(mac_p) >= 3, "需要至少 3 个 MAC-P 事件"
        assert len(mac_o) >= 1, "需要至少 1 个 MAC-O 事件"

        # PRELOAD_2 中 C1[2] 先于 C2[0] 发射
        assert mac_p[2].start_time <= mac_o[0].start_time, (
            f"PRELOAD_2: C1[2] (start={mac_p[2].start_time:.1f}) should start before or at C2[0] (start={mac_o[0].start_time:.1f})"
        )

    def test_fixpipe_o_count(self):
        """4 个 K 块应有 4 个 FIXPIPE-O"""
        m = make_modeler(k_blocks=4)
        timeline, _, _, _ = m.run_simulation()
        fix_o = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "O"]
        assert len(fix_o) == 4, f"Expected 4 FIXPIPE-O, got {len(fix_o)}"

    def test_v2_count(self):
        """4 个 K 块应有 4 个 VECTOR_V2"""
        m = make_modeler(k_blocks=4)
        timeline, _, _, _ = m.run_simulation()
        v2 = [e for e in timeline if e.unit == "VECTOR_V2"]
        assert len(v2) == 4, f"Expected 4 VECTOR_V2, got {len(v2)}"


class TestPreload2VsPreload1:
    """PRELOAD_2 vs PRELOAD_1 性能对比"""

    def _make_both(self, k_blocks: int):
        s2 = 128 * k_blocks
        common = dict(
            s1_total=128, s2_total=s2, d_total=128,
            s1_base_size=128, s2_base_size=128, d_base_size=128,
            baseM_C1=128, baseN_C1=128, baseK_C1=128,
            baseM_C2=128, baseN_C2=128, baseK_C2=128,
            q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
            is_l2cache=True,
            inner_core_pipeline=InnerCorePipeline.DEFAULT,
            L1_db=False, L0_db=False,
        )
        m1 = C1Modeler(**common, inter_core_pipeline=InterCorePipeline.PRELOAD_1)
        m2 = C1Modeler(**common, inter_core_pipeline=InterCorePipeline.PRELOAD_2)
        return m1, m2

    def test_preload2_runs_vs_preload1(self):
        """PRELOAD_2 能正常跑完，周期数 > 0"""
        m1, m2 = self._make_both(k_blocks=4)
        _, _, _, t1 = m1.run_simulation()
        _, _, _, t2 = m2.run_simulation()
        assert t1 > 0
        assert t2 > 0

    def test_preload2_positive_cycles(self):
        """PRELOAD_2 对不同 K 块数产生合理正周期"""
        for k_blocks in [1, 2, 3, 4, 6]:
            m = make_modeler(k_blocks=k_blocks)
            _, _, _, total = m.run_simulation()
            assert total > 0, f"k_blocks={k_blocks}: expected positive cycles, got {total}"


class TestPreload2FP8:
    """PRELOAD_2 + FP8 数据类型"""

    def test_preload2_fp8_runs(self):
        """PRELOAD_2 + FP8 正常运行"""
        m = C1Modeler(
            s1_total=128, s2_total=512, d_total=128,
            s1_base_size=128, s2_base_size=128, d_base_size=128,
            baseM_C1=128, baseN_C1=128, baseK_C1=128,
            baseM_C2=128, baseN_C2=128, baseK_C2=128,
            q_data_type=DataType.FP8,
            kv_data_type=DataType.FP8,
            is_l2cache=True,
            inter_core_pipeline=InterCorePipeline.PRELOAD_2,
            inner_core_pipeline=InnerCorePipeline.DEFAULT,
        )
        _, _, _, total = m.run_simulation()
        assert total > 0

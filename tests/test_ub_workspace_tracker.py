"""
L1PSlotTracker 和 UBSlotTracker 单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from modelers.c1_modeler import L1PSlotTracker, UBSlotTracker


class TestL1PSlotTrackerBasic:
    """L1PSlotTracker 基础功能测试"""

    def test_init_2slots(self):
        """2槽 初始化"""
        tracker = L1PSlotTracker(n_slots=2)
        assert tracker.n_slots == 2
        assert tracker.slot_free_times == [0.0, 0.0]

    def test_init_3slots(self):
        """3槽 初始化"""
        tracker = L1PSlotTracker(n_slots=3)
        assert tracker.n_slots == 3
        assert tracker.slot_free_times == [0.0, 0.0, 0.0]

    def test_allocate_returns_slot_idx_and_time(self):
        """allocate 返回 (slot_idx, actual_avail) 元组"""
        tracker = L1PSlotTracker(n_slots=2)
        slot_idx, actual_avail = tracker.allocate(0.0)
        assert isinstance(slot_idx, int)
        assert isinstance(actual_avail, float)
        assert 0 <= slot_idx < 2

    def test_allocate_free_slot_returns_desired(self):
        """全空时 actual_avail == desired"""
        tracker = L1PSlotTracker(n_slots=2)
        _, actual_avail = tracker.allocate(100.0)
        assert actual_avail == 100.0

    def test_release_updates_slot_free_time(self):
        """release 更新对应槽的空闲时间"""
        tracker = L1PSlotTracker(n_slots=2)
        slot_idx, _ = tracker.allocate(0.0)
        tracker.release(slot_idx, 500.0)
        assert tracker.slot_free_times[slot_idx] == 500.0


class TestL1PSlotTrackerWaiting:
    """L1PSlotTracker 等待逻辑测试"""

    def test_single_slot_must_wait(self):
        """单槽：MTE1-P 未完成时，新 MTE3 必须等待"""
        tracker = L1PSlotTracker(n_slots=1)
        s0, _ = tracker.allocate(0.0)
        tracker.release(s0, 1000.0)  # MTE1-P 在 1000 结束，槽释放

        s1, actual_avail = tracker.allocate(0.0)  # 下一轮期望从 0 开始
        assert actual_avail == 1000.0  # 必须等到 1000

    def test_2slots_both_allocate_no_wait(self):
        """2槽：连续两次 allocate（无 release）均不需要等待"""
        tracker = L1PSlotTracker(n_slots=2)
        _, t0 = tracker.allocate(0.0)
        assert t0 == 0.0

        _, t1 = tracker.allocate(0.0)
        assert t1 == 0.0

    def test_2slots_after_release_picks_released_slot(self):
        """2槽：release 一个槽后，allocate 选择更早空闲的槽"""
        tracker = L1PSlotTracker(n_slots=2)
        s0, _ = tracker.allocate(0.0)
        tracker.release(s0, 400.0)  # slot0 free=400

        # 此时 slot0 free=400, slot1 free=0 → 应选 slot1（更早空闲）
        s_new, actual_avail = tracker.allocate(0.0)
        assert actual_avail == 0.0  # slot1 空闲，no wait
        assert s_new != s0

    def test_2slots_both_busy_waits_earlier(self):
        """2槽：两个槽均有 release 后，选择最早释放的"""
        tracker = L1PSlotTracker(n_slots=2)
        s0, _ = tracker.allocate(0.0)
        tracker.release(s0, 400.0)
        s1, _ = tracker.allocate(0.0)
        tracker.release(s1, 600.0)

        s_new, actual_avail = tracker.allocate(0.0)
        assert actual_avail == 400.0
        assert s_new == s0

    def test_3slots_allocates_no_wait(self):
        """3槽：三次连续 allocate 均不需要等待"""
        tracker = L1PSlotTracker(n_slots=3)
        for desired in [0.0, 50.0, 100.0]:
            _, actual = tracker.allocate(desired)
            assert actual == desired  # 无等待

    def test_3slots_fourth_alloc_waits_after_all_released(self):
        """3槽：三个槽全部 release 后，第 4 次 allocate 等待最早的"""
        tracker = L1PSlotTracker(n_slots=3)
        s0, _ = tracker.allocate(0.0)
        tracker.release(s0, 300.0)
        s1, _ = tracker.allocate(0.0)
        tracker.release(s1, 500.0)
        s2, _ = tracker.allocate(0.0)
        tracker.release(s2, 700.0)

        s_new, actual_avail = tracker.allocate(0.0)
        assert actual_avail == 300.0  # 等最早释放的 s0
        assert s_new == s0

    def test_desired_after_free_no_wait(self):
        """desired_start 比槽空闲时间晚，actual == desired"""
        tracker = L1PSlotTracker(n_slots=1)
        s0, _ = tracker.allocate(0.0)
        tracker.release(s0, 100.0)

        _, actual = tracker.allocate(200.0)
        assert actual == 200.0

    def test_desired_before_free_must_wait(self):
        """desired_start 比槽空闲时间早，actual == 槽空闲时间"""
        tracker = L1PSlotTracker(n_slots=1)
        s0, _ = tracker.allocate(0.0)
        tracker.release(s0, 500.0)

        _, actual = tracker.allocate(200.0)
        assert actual == 500.0


class TestUBSlotTrackerBasic:
    """UBSlotTracker 基础功能测试（固定2槽）"""

    def test_init_always_2slots(self):
        """始终 2 个槽"""
        tracker = UBSlotTracker()
        assert tracker.slot_free_times == [0.0, 0.0]

    def test_allocate_returns_slot_idx_and_time(self):
        """allocate 返回 (slot_idx, actual_avail)"""
        tracker = UBSlotTracker()
        slot_idx, actual_avail = tracker.allocate(0.0)
        assert isinstance(slot_idx, int)
        assert 0 <= slot_idx < 2
        assert actual_avail == 0.0

    def test_two_allocs_no_wait(self):
        """两次连续 allocate 都不需要等待"""
        tracker = UBSlotTracker()
        _, t0 = tracker.allocate(0.0)
        _, t1 = tracker.allocate(0.0)
        assert t0 == 0.0
        assert t1 == 0.0

    def test_release_then_third_alloc_waits(self):
        """两槽都被占用并 release 后，第三次 allocate 等最早释放的"""
        tracker = UBSlotTracker()
        s0, _ = tracker.allocate(0.0)
        tracker.release(s0, 400.0)
        s1, _ = tracker.allocate(0.0)
        tracker.release(s1, 600.0)

        s_new, actual_avail = tracker.allocate(0.0)
        assert actual_avail == 400.0
        assert s_new == s0

    def test_mte3_releases_ub_c1_slot(self):
        """模拟 C1 侧：FIXPIPE-P 占用槽，MTE3 完成后释放，下一个 FIXPIPE-P 可用"""
        tracker = UBSlotTracker()

        # k0: FIXPIPE-P@100 占用，MTE3@500 完成释放
        slot0, avail0 = tracker.allocate(100.0)
        assert avail0 == 100.0
        tracker.release(slot0, 500.0)  # MTE3 完成

        # k1: 另一槽还空闲，无需等待
        slot1, avail1 = tracker.allocate(100.0)
        assert avail1 == 100.0  # 另一槽 free_time=0 < desired=100，返回 desired

        tracker.release(slot1, 700.0)  # MTE3 完成

        # k2: 两槽 free=[500, 700]，选最早的 slot0
        _, avail2 = tracker.allocate(0.0)
        assert avail2 == 500.0  # 必须等 slot0 空闲

    def test_v2_releases_ub_c2_slot(self):
        """模拟 C2 侧：FIXPIPE-O 占用槽，V2 完成后释放，下一个 FIXPIPE-O 可用"""
        tracker = UBSlotTracker()

        # k0: FIXPIPE-O@200，V2@800 完成释放
        slot0, avail0 = tracker.allocate(200.0)
        assert avail0 == 200.0
        tracker.release(slot0, 800.0)

        # k1: 另一槽空闲
        slot1, avail1 = tracker.allocate(200.0)
        assert avail1 == 200.0

        tracker.release(slot1, 1000.0)

        # k2: 等最早释放
        _, avail2 = tracker.allocate(0.0)
        assert avail2 == 800.0


class TestL1PSlotTrackerSequential:
    """L1PSlotTracker 顺序分配场景测试"""

    def test_2slots_pipeline_key_constraint(self):
        """
        2槽流水约束：验证 MTE3 受前序 MTE1-P 释放约束。
        模拟: k0,k1 (allocate) → k0 MTE1-P done (release @400)
              → k2 picks free slot → k1 MTE1-P done (release @500)
              → k3: both released, waits for min(400,500)=400
        """
        tracker = L1PSlotTracker(n_slots=2)
        slot_indices = []

        s0, avail0 = tracker.allocate(0.0)
        slot_indices.append(s0)
        assert avail0 == 0.0

        s1, avail1 = tracker.allocate(0.0)
        slot_indices.append(s1)
        assert avail1 == 0.0

        # k0 MTE1-P 完成，释放槽
        tracker.release(slot_indices[0], 400.0)

        # k2: 有槽空闲（slot1 free=0），无等待
        s2, avail2 = tracker.allocate(0.0)
        slot_indices.append(s2)
        assert avail2 == 0.0

        # k1 MTE1-P 完成
        tracker.release(slot_indices[1], 500.0)
        # k2 MTE1-P 完成（重用了 slot0）
        tracker.release(slot_indices[2], 600.0)

        # k3: 两槽均释放，等最早的
        _, avail3 = tracker.allocate(0.0)
        assert avail3 == 500.0  # min(500, 600) = 500（slot1）

    def test_3slots_pipeline_constraint_fires(self):
        """3槽流水：所有槽 release 后，第 4 次 allocate 等最早的。"""
        tracker = L1PSlotTracker(n_slots=3)

        s0, _ = tracker.allocate(0.0)
        assert s0 == 0
        tracker.release(s0, 300.0)

        s1, _ = tracker.allocate(0.0)
        assert s1 == 1
        tracker.release(s1, 500.0)

        s2, _ = tracker.allocate(0.0)
        assert s2 == 2
        tracker.release(s2, 700.0)

        _, avail_next = tracker.allocate(0.0)
        assert avail_next == 300.0  # 等最早释放的 slot0

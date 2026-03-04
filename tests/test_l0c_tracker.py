"""
L0CSlotTracker 单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from modelers.c1_modeler import L0CSlotTracker


class TestL0CSlotTrackerBasic:
    """基础功能测试"""

    def test_init_single_slot(self):
        """单槽初始化：空闲时间均为 0"""
        tracker = L0CSlotTracker(n_slots=1)
        assert tracker.n_slots == 1
        assert tracker.slot_free_times == [0.0]

    def test_init_doublebuffer(self):
        """双缓冲初始化：2 个槽，空闲时间均为 0"""
        tracker = L0CSlotTracker(n_slots=2)
        assert tracker.n_slots == 2
        assert tracker.slot_free_times == [0.0, 0.0]

    def test_allocate_returns_slot_and_time(self):
        """allocate 返回 (slot_idx, actual_start) 元组"""
        tracker = L0CSlotTracker(n_slots=2)
        slot_idx, actual_start = tracker.allocate(100.0)
        assert isinstance(slot_idx, int)
        assert isinstance(actual_start, float)
        assert 0 <= slot_idx < 2

    def test_allocate_first_slot_free(self):
        """全空时 allocate 立即返回，actual_start == desired_start"""
        tracker = L0CSlotTracker(n_slots=2)
        _, actual_start = tracker.allocate(50.0)
        assert actual_start == 50.0

    def test_release_updates_slot_free_time(self):
        """release 后，该槽的空闲时间被更新"""
        tracker = L0CSlotTracker(n_slots=2)
        slot_idx, _ = tracker.allocate(0.0)
        tracker.release(slot_idx, 200.0)
        assert tracker.slot_free_times[slot_idx] == 200.0


class TestL0CSlotTrackerWaiting:
    """等待逻辑测试"""

    def test_must_wait_when_slot_busy(self):
        """槽被占用时，actual_start >= 槽的空闲时间"""
        tracker = L0CSlotTracker(n_slots=1)
        slot0, _ = tracker.allocate(0.0)
        tracker.release(slot0, 500.0)

        # 第二次 allocate desired=100，但槽要到 500 才释放
        slot1, actual_start = tracker.allocate(100.0)
        assert actual_start == 500.0

    def test_doublebuffer_no_wait_if_other_slot_free(self):
        """双缓冲：一个槽忙但另一个空闲时，应选空闲槽（无需等待）"""
        tracker = L0CSlotTracker(n_slots=2)
        slot0, _ = tracker.allocate(0.0)
        tracker.release(slot0, 1000.0)  # slot0 占用到 1000

        # slot1 依然空闲，allocate 应选 slot1，actual_start == desired
        slot_new, actual_start = tracker.allocate(50.0)
        assert actual_start == 50.0
        assert slot_new != slot0  # 应选另一槽

    def test_doublebuffer_both_busy_waits_earlier(self):
        """双缓冲：两个槽均忙时，选择最早释放的槽"""
        tracker = L0CSlotTracker(n_slots=2)
        s0, _ = tracker.allocate(0.0)
        tracker.release(s0, 300.0)
        s1, _ = tracker.allocate(0.0)
        tracker.release(s1, 500.0)

        # 两槽都忙，desired=0，应等最早的槽（300）
        slot_new, actual_start = tracker.allocate(0.0)
        assert actual_start == 300.0
        assert slot_new == s0

    def test_desired_start_gt_free_time_no_wait(self):
        """desired_start 比槽空闲时间更晚时，actual_start == desired_start"""
        tracker = L0CSlotTracker(n_slots=1)
        s0, _ = tracker.allocate(0.0)
        tracker.release(s0, 100.0)

        _, actual_start = tracker.allocate(200.0)
        assert actual_start == 200.0  # desired 更晚，直接用 desired

    def test_desired_start_lt_free_time_waits(self):
        """desired_start 比槽空闲时间更早时，actual_start == 槽空闲时间"""
        tracker = L0CSlotTracker(n_slots=1)
        s0, _ = tracker.allocate(0.0)
        tracker.release(s0, 300.0)

        _, actual_start = tracker.allocate(100.0)
        assert actual_start == 300.0  # desired 更早，必须等待到 300


class TestL0CSlotTrackerSequential:
    """顺序分配场景测试"""

    def test_sequential_two_alloc_release(self):
        """顺序使用同一槽：alloc → release → alloc → release"""
        tracker = L0CSlotTracker(n_slots=1)
        s0, t0 = tracker.allocate(0.0)
        assert t0 == 0.0
        tracker.release(s0, 100.0)

        s1, t1 = tracker.allocate(0.0)
        assert t1 == 100.0  # 必须等前次释放
        tracker.release(s1, 200.0)

        s2, t2 = tracker.allocate(0.0)
        assert t2 == 200.0

    def test_doublebuffer_interleaved(self):
        """双缓冲：先 release slot0，再 allocate 时得到更早空闲的槽"""
        tracker = L0CSlotTracker(n_slots=2)
        # MAC1 用 slot0，从 0 开始
        s0, t0 = tracker.allocate(0.0)
        assert t0 == 0.0

        # FIXPIPE0 完成，释放 slot0 到 150
        tracker.release(s0, 150.0)
        # 此时 slot0 free=150, slot1 free=0 → 选 slot1（更早空闲）
        s1, t1 = tracker.allocate(50.0)
        assert t1 == 50.0  # slot1 free=0 < 50，取 desired=50
        assert s1 != s0    # 应选 slot1

        # FIXPIPE1 完成，释放 slot1 到 200
        tracker.release(s1, 200.0)
        # 此时 slot0 free=150, slot1 free=200 → 选 slot0
        s2, t2 = tracker.allocate(100.0)
        assert t2 == 150.0  # slot0 free=150 > desired=100，等待
        assert s2 == s0     # 应选 slot0（free_time 更小）

    def test_many_alloc_release_cycles(self):
        """多轮 alloc/release 验证时间单调递增"""
        tracker = L0CSlotTracker(n_slots=2)
        release_time = 0.0
        for i in range(10):
            desired = release_time
            slot_idx, actual = tracker.allocate(desired)
            assert actual >= desired
            release_time = actual + 100.0
            tracker.release(slot_idx, release_time)

"""
UBWorkspaceTracker 单元测试
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from modelers.c1_modeler import UBWorkspaceTracker


class TestUBWorkspaceTrackerBasic:
    """基础功能测试"""

    def test_init_2ws(self):
        """2WS 初始化"""
        tracker = UBWorkspaceTracker(n_ws=2)
        assert tracker.n_ws == 2
        assert tracker.ws_free_times == [0.0, 0.0]

    def test_init_3ws(self):
        """3WS 初始化"""
        tracker = UBWorkspaceTracker(n_ws=3)
        assert tracker.n_ws == 3
        assert tracker.ws_free_times == [0.0, 0.0, 0.0]

    def test_allocate_returns_ws_idx_and_time(self):
        """allocate 返回 (ws_idx, actual_avail) 元组"""
        tracker = UBWorkspaceTracker(n_ws=2)
        ws_idx, actual_avail = tracker.allocate(0.0)
        assert isinstance(ws_idx, int)
        assert isinstance(actual_avail, float)
        assert 0 <= ws_idx < 2

    def test_allocate_free_ws_returns_desired(self):
        """全空时 actual_avail == desired"""
        tracker = UBWorkspaceTracker(n_ws=2)
        _, actual_avail = tracker.allocate(100.0)
        assert actual_avail == 100.0

    def test_release_updates_ws_free_time(self):
        """release 更新对应 WS 的空闲时间"""
        tracker = UBWorkspaceTracker(n_ws=2)
        ws_idx, _ = tracker.allocate(0.0)
        tracker.release(ws_idx, 500.0)
        assert tracker.ws_free_times[ws_idx] == 500.0


class TestUBWorkspaceTrackerWaiting:
    """等待逻辑测试"""

    def test_single_ws_must_wait(self):
        """单 WS：前次 V2 未完成时，新 MTE2 必须等待"""
        tracker = UBWorkspaceTracker(n_ws=1)
        ws0, _ = tracker.allocate(0.0)
        tracker.release(ws0, 1000.0)  # V2 在 1000 结束

        ws1, actual_avail = tracker.allocate(0.0)  # 下一轮期望从 0 开始
        assert actual_avail == 1000.0  # 必须等到 1000

    def test_2ws_both_allocate_no_wait(self):
        """2WS：连续两次 allocate（无 release）均不需要等待"""
        tracker = UBWorkspaceTracker(n_ws=2)
        _, t0 = tracker.allocate(0.0)
        assert t0 == 0.0

        _, t1 = tracker.allocate(0.0)
        assert t1 == 0.0  # 第二次也不需要等待（free_time 相同，选同一槽也 ok）

    def test_2ws_after_release_picks_released_slot(self):
        """2WS：release 一个 WS 后，allocate 选择更早空闲的槽"""
        tracker = UBWorkspaceTracker(n_ws=2)
        ws0, _ = tracker.allocate(0.0)
        tracker.release(ws0, 400.0)  # slot0 free=400

        # 此时 slot0 free=400, slot1 free=0 → 应选 slot1（更早空闲）
        ws_new, actual_avail = tracker.allocate(0.0)
        assert actual_avail == 0.0  # slot1 空闲，no wait
        assert ws_new != ws0        # 应选 slot1

    def test_2ws_both_busy_waits_earlier(self):
        """2WS：两个 WS 均有 release 后，选择最早释放的"""
        tracker = UBWorkspaceTracker(n_ws=2)
        ws0, _ = tracker.allocate(0.0)
        tracker.release(ws0, 400.0)
        ws1, _ = tracker.allocate(0.0)
        tracker.release(ws1, 600.0)

        ws_new, actual_avail = tracker.allocate(0.0)
        assert actual_avail == 400.0
        assert ws_new == ws0

    def test_3ws_allocates_no_wait(self):
        """3WS：三次连续 allocate 均不需要等待"""
        tracker = UBWorkspaceTracker(n_ws=3)
        for desired in [0.0, 50.0, 100.0]:
            _, actual = tracker.allocate(desired)
            assert actual == desired  # 无等待

    def test_3ws_fourth_alloc_waits_after_all_released(self):
        """3WS：三个 WS 全部 release 后，第 4 次 allocate 等待最早的"""
        tracker = UBWorkspaceTracker(n_ws=3)
        ws0, _ = tracker.allocate(0.0)
        tracker.release(ws0, 300.0)
        ws1, _ = tracker.allocate(0.0)
        tracker.release(ws1, 500.0)
        ws2, _ = tracker.allocate(0.0)
        tracker.release(ws2, 700.0)

        ws_new, actual_avail = tracker.allocate(0.0)
        assert actual_avail == 300.0  # 等最早释放的 ws0
        assert ws_new == ws0

    def test_desired_after_free_no_wait(self):
        """desired_start 比 WS 空闲时间晚，actual == desired"""
        tracker = UBWorkspaceTracker(n_ws=1)
        ws0, _ = tracker.allocate(0.0)
        tracker.release(ws0, 100.0)

        _, actual = tracker.allocate(200.0)
        assert actual == 200.0

    def test_desired_before_free_must_wait(self):
        """desired_start 比 WS 空闲时间早，actual == WS 空闲时间"""
        tracker = UBWorkspaceTracker(n_ws=1)
        ws0, _ = tracker.allocate(0.0)
        tracker.release(ws0, 500.0)

        _, actual = tracker.allocate(200.0)
        assert actual == 500.0


class TestUBWorkspaceTrackerSequential:
    """顺序分配场景测试（与生产代码使用模式一致）"""

    def test_2ws_pipeline_key_constraint(self):
        """
        2WS 流水约束：一旦所有 WS 被 release，下次 allocate 等待最早释放。
        模拟: k0,k1 (allocate) → k0 V2 done (release 0 @400)
              → k2 picks free slot → k1 V2 done (release 0 @500)
              → k3 picks free slot → k2 V2 done (release 1 @600)
              → k4: both released, waits for min(500, 600)=500
        """
        tracker = UBWorkspaceTracker(n_ws=2)
        ws_indices = []

        # k0, k1: no constraint yet
        ws0, avail0 = tracker.allocate(0.0)
        ws_indices.append(ws0)
        assert avail0 == 0.0

        ws1, avail1 = tracker.allocate(0.0)
        ws_indices.append(ws1)
        assert avail1 == 0.0

        # k0 V2 done: release slot
        tracker.release(ws_indices[0], 400.0)

        # k2: picks newly freed slot (other slot still fresh)
        ws2, avail2 = tracker.allocate(0.0)
        ws_indices.append(ws2)
        assert avail2 == 0.0  # some slot is still free

        # k1 V2 done: release slot (same as ws_indices[1])
        tracker.release(ws_indices[1], 500.0)

        # k3: some slot still free
        ws3, avail3 = tracker.allocate(0.0)
        ws_indices.append(ws3)
        assert avail3 == 0.0

        # k2 V2 done
        tracker.release(ws_indices[2], 600.0)
        # k3 V2 done
        tracker.release(ws_indices[3], 700.0)

        # k4: now all slots have been released → must wait
        _, avail4 = tracker.allocate(0.0)
        assert avail4 > 0.0  # 必须等待某个 WS

    def test_3ws_pipeline_constraint_fires(self):
        """3WS 流水：在所有 3 个 WS 被 release 后，下次 allocate 等待最早的。
        需要 release 穿插 allocate，确保每次 allocate 拿到不同槽。"""
        tracker = UBWorkspaceTracker(n_ws=3)

        # 第一次：slot0 空闲，分配 slot0，立即 release@300
        ws0, _ = tracker.allocate(0.0)  # → slot0
        assert ws0 == 0
        tracker.release(ws0, 300.0)  # [300, 0, 0]

        # 第二次：slot1/2 更早 (0.0 < 300)，分配 slot1，立即 release@500
        ws1, _ = tracker.allocate(0.0)  # → slot1 (0.0 < 300)
        assert ws1 == 1
        tracker.release(ws1, 500.0)  # [300, 500, 0]

        # 第三次：slot2 更早 (0.0 < 300 < 500)，分配 slot2，立即 release@700
        ws2, _ = tracker.allocate(0.0)  # → slot2 (0.0 < 300 < 500)
        assert ws2 == 2
        tracker.release(ws2, 700.0)  # [300, 500, 700]

        # 第四次：所有槽均有 release，min=300 → 等待
        _, avail_next = tracker.allocate(0.0)
        assert avail_next == 300.0  # 等最早释放的 slot0

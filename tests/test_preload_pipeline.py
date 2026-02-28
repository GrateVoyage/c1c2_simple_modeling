# tests/test_preload_pipeline.py
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(pipeline, two_buffer=False):
    return C1Modeler(
        s1_total=128, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=pipeline,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
        L1_db=False, L0_db=False, two_buffer=two_buffer,
    )

def test_preload_runs():
    m = make_modeler(InterCorePipeline.PRELOAD)
    _, _, _, total = m.run_simulation()
    assert total > 0

def test_preload_v_loaded_before_v1_ends():
    """
    PRELOAD core: V should be in L1 before V1 completes.
    The first MTE2-V event should end BEFORE the first VECTOR_V1 event ends.
    """
    m = make_modeler(InterCorePipeline.PRELOAD)
    timeline, _, _, _ = m.run_simulation()
    mte2_v = [e for e in timeline if e.unit == "MTE2" and "V" in e.operation]
    v1_events = [e for e in timeline if e.unit == "VECTOR_V1"]
    assert len(mte2_v) > 0, "Should have MTE2 V events"
    assert len(v1_events) > 0, "Should have VECTOR_V1 events"
    assert mte2_v[0].end_time < v1_events[0].end_time, (
        f"V should be loaded before V1 ends: V end={mte2_v[0].end_time:.1f}, "
        f"V1 end={v1_events[0].end_time:.1f}"
    )

def test_preload_no_mte2_v_in_c2_phase():
    """
    In PRELOAD mode, C2 phase should NOT have a separate MTE2 V load.
    Total MTE2-V events == number of k blocks (one V load per k block, done during C1).
    For 1 q_block x 2 k_blocks: exactly 2 MTE2-V events total.
    """
    m = make_modeler(InterCorePipeline.PRELOAD)
    timeline, _, _, _ = m.run_simulation()
    mte2_v = [e for e in timeline if e.unit == "MTE2" and "V" in e.operation]
    k_block_count = 512 // 256  # = 2
    assert len(mte2_v) == k_block_count, (
        f"Expected {k_block_count} MTE2-V events (one per k block, during C1 phase), "
        f"got {len(mte2_v)}"
    )

def test_preload_faster_than_default():
    """PRELOAD should be faster than DEFAULT (V preloading reduces C2 wait time)"""
    m_default = make_modeler(InterCorePipeline.DEFAULT)
    m_preload = make_modeler(InterCorePipeline.PRELOAD)
    _, _, _, t_default = m_default.run_simulation()
    _, _, _, t_preload = m_preload.run_simulation()
    assert t_preload < t_default, (
        f"PRELOAD should be faster than DEFAULT: "
        f"preload={t_preload:.0f}, default={t_default:.0f}"
    )

def test_preload_two_buffer_runs():
    """PRELOAD with two_buffer=True should also work"""
    m = make_modeler(InterCorePipeline.PRELOAD, two_buffer=True)
    _, _, _, total = m.run_simulation()
    assert total > 0

def test_preload_two_buffer_faster_than_no_two_buffer():
    """TWOBUFFER eliminates V slot contention, so it should be faster or equal"""
    m_no_tb = make_modeler(InterCorePipeline.PRELOAD, two_buffer=False)
    m_tb = make_modeler(InterCorePipeline.PRELOAD, two_buffer=True)
    _, _, _, t_no_tb = m_no_tb.run_simulation()
    _, _, _, t_tb = m_tb.run_simulation()
    assert t_tb <= t_no_tb, (
        f"TWOBUFFER should not be slower: two_buffer={t_tb:.0f}, no_two_buffer={t_no_tb:.0f}"
    )

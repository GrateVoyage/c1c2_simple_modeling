# tests/test_preload_pipeline.py
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(pipeline):
    return C1Modeler(
        s1_total=128, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=pipeline,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
        L1_db=False, L0_db=False,
    )

def test_preload_runs():
    m = make_modeler(InterCorePipeline.PRELOAD)
    _, _, _, total = m.run_simulation()
    assert total > 0

def test_preload_v_loaded_in_c2_phase():
    """
    PRELOAD core: V should be loaded during C2 phase (after V1 ends).
    MTE2-V events should start AFTER the corresponding VECTOR_V1 ends.
    """
    m = make_modeler(InterCorePipeline.PRELOAD)
    timeline, _, _, _ = m.run_simulation()
    mte2_v = [e for e in timeline if e.unit == "MTE2" and "V" in e.operation]
    v1_events = [e for e in timeline if e.unit == "VECTOR_V1"]
    assert len(mte2_v) > 0, "Should have MTE2 V events"
    assert len(v1_events) > 0, "Should have VECTOR_V1 events"
    # In PRELOAD mode, V loads in C2 (after p_ready_time which is end of V1)
    # The first MTE2-V event starts at or after the first V1 ends
    assert mte2_v[0].start_time >= v1_events[0].end_time, (
        f"V should be loaded after V1 ends (C2 phase): "
        f"MTE2-V start={mte2_v[0].start_time:.1f}, V1 end={v1_events[0].end_time:.1f}"
    )

def test_preload_mte2_v_count():
    """
    In PRELOAD mode, V is loaded in C2 phase (same as DEFAULT).
    Total MTE2-V events == number of k blocks.
    For 1 q_block x 2 k_blocks: exactly 2 MTE2-V events total.
    """
    m = make_modeler(InterCorePipeline.PRELOAD)
    timeline, _, _, _ = m.run_simulation()
    mte2_v = [e for e in timeline if e.unit == "MTE2" and "V" in e.operation]
    k_block_count = 512 // 256  # = 2
    assert len(mte2_v) == k_block_count, (
        f"Expected {k_block_count} MTE2-V events (one per k block in C2 phase), "
        f"got {len(mte2_v)}"
    )

def test_preload_faster_than_default():
    """PRELOAD should be faster than DEFAULT (C1 overlaps with previous C2)"""
    m_default = make_modeler(InterCorePipeline.DEFAULT)
    m_preload = make_modeler(InterCorePipeline.PRELOAD)
    _, _, _, t_default = m_default.run_simulation()
    _, _, _, t_preload = m_preload.run_simulation()
    assert t_preload < t_default, (
        f"PRELOAD should be faster than DEFAULT: "
        f"preload={t_preload:.0f}, default={t_default:.0f}"
    )

def test_preload_c1_overlaps_with_previous_c2():
    """
    In PRELOAD mode, C1[k+1] MAC should start before C2[k] completes.
    Verify that the second MAC-P event starts before the first MAC-O event ends.
    """
    m = make_modeler(InterCorePipeline.PRELOAD)
    timeline, _, _, _ = m.run_simulation()
    mac_p_events = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    mac_o_events = [e for e in timeline if e.unit == "MAC" and e.operation == "O"]
    k_block_count = 512 // 256  # = 2
    if k_block_count >= 2:
        # C1[1] MAC should start before C2[0] MAC ends (or at same time)
        # This validates pipeline overlap
        assert len(mac_p_events) >= 2, "Need at least 2 MAC-P events"
        assert len(mac_o_events) >= 1, "Need at least 1 MAC-O event"
        # C1[1] starts before or when C2[0] starts (pipeline overlap)
        assert mac_p_events[1].start_time <= mac_o_events[0].start_time, (
            f"PRELOAD: C1[1] MAC should start no later than C2[0] MAC: "
            f"C1[1] start={mac_p_events[1].start_time:.1f}, C2[0] start={mac_o_events[0].start_time:.1f}"
        )

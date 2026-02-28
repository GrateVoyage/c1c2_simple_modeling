# tests/test_n_buffer.py
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(pipeline=InterCorePipeline.N_BUFFER, **kwargs):
    defaults = dict(
        s1_total=128, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=pipeline,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
        L1_db=False, L0_db=False,
    )
    defaults.update(kwargs)
    return C1Modeler(**defaults)

def test_n_buffer_runs_without_error():
    m = make_modeler()
    timeline, bound_type, unit_times, total_cycles = m.run_simulation()
    assert total_cycles > 0
    assert len(timeline) > 0

def test_n_buffer_has_all_units():
    m = make_modeler()
    timeline, _, _, _ = m.run_simulation()
    units = {e.unit for e in timeline}
    assert "MAC" in units
    assert "VECTOR_V1" in units
    assert "VECTOR_V2" in units
    assert "MTE2" in units
    assert "MTE1" in units
    assert "FIXPIPE" in units
    assert "MTE3" in units

def test_n_buffer_correct_event_count():
    """
    1 q_block × 2 k_blocks:
    - 2 MAC P events (C1 × 2k)
    - 2 VECTOR_V1 events
    - 2 MAC O events (C2 × 2k)
    - 2 VECTOR_V2 events
    """
    m = make_modeler(s1_total=128, s2_total=512,
                     s1_base_size=128, s2_base_size=256)
    timeline, _, _, _ = m.run_simulation()
    mac_c1 = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    mac_c2 = [e for e in timeline if e.unit == "MAC" and e.operation == "O"]
    v1 = [e for e in timeline if e.unit == "VECTOR_V1"]
    v2 = [e for e in timeline if e.unit == "VECTOR_V2"]
    assert len(mac_c1) == 2
    assert len(mac_c2) == 2
    assert len(v1) == 2
    assert len(v2) == 2

def test_n_buffer_v1_overlaps_mac_c1():
    """
    N_BUFFER key feature: VECTOR_V1[k0] should start BEFORE MAC C1[k1] ends.
    They run on different hardware (VECTOR vs MAC), so they can overlap.
    """
    m = make_modeler(s1_total=128, s2_total=512,
                     s1_base_size=128, s2_base_size=256)
    timeline, _, _, _ = m.run_simulation()
    mac_c1 = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    v1 = [e for e in timeline if e.unit == "VECTOR_V1"]
    if len(mac_c1) >= 2 and len(v1) >= 1:
        # V1[k0] starts before MAC C1[k1] ends
        assert v1[0].start_time < mac_c1[1].end_time, (
            f"V1[k0] start={v1[0].start_time:.1f} should be < "
            f"MAC_C1[k1] end={mac_c1[1].end_time:.1f}"
        )

# tests/test_c1_data_sizes.py
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(**kwargs):
    defaults = dict(
        s1_total=128, s2_total=256, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    defaults.update(kwargs)
    return C1Modeler(**defaults)

def test_fixpipe_p_is_fp32():
    """FIXPIPE P size = s1_base * s2_base * 4 (FP32), regardless of data type"""
    m = make_modeler(kv_data_type=DataType.FP16)
    assert m._calc_fixpipe_p_size() == 128 * 256 * 4  # 131072

def test_fixpipe_p_is_fp32_also_with_fp8():
    """FIXPIPE P size does NOT change with FP8 — still FP32"""
    m = make_modeler(kv_data_type=DataType.FP8)
    assert m._calc_fixpipe_p_size() == 128 * 256 * 4  # still 131072

def test_fixpipe_o_is_fp32():
    """FIXPIPE O size = s1_base * d_base * 4 (FP32)"""
    m = make_modeler()
    assert m._calc_fixpipe_o_size() == 128 * 128 * 4  # 65536

def test_mte3_p_uses_kv_dtype_fp16():
    """MTE3 P size = s1_base * s2_base * kv_elem_size"""
    m = make_modeler(kv_data_type=DataType.FP16)
    assert m._calc_mte3_p_size() == 128 * 256 * 2  # 65536

def test_mte3_p_uses_kv_dtype_fp8():
    m = make_modeler(kv_data_type=DataType.FP8)
    assert m._calc_mte3_p_size() == 128 * 256 * 1  # 32768

def test_q_size_uses_q_dtype_fp16():
    """Q size = s1_base * d_base * q_elem_size"""
    m = make_modeler(q_data_type=DataType.FP16)
    assert m._calc_q_size() == 128 * 128 * 2  # 32768

def test_q_size_uses_q_dtype_fp8():
    m = make_modeler(q_data_type=DataType.FP8)
    assert m._calc_q_size() == 128 * 128 * 1  # 16384

def test_k_size_uses_kv_dtype_fp16():
    """K size = s2_base * d_base * kv_elem_size"""
    m = make_modeler(kv_data_type=DataType.FP16)
    assert m._calc_k_size() == 256 * 128 * 2  # 65536

def test_k_size_uses_kv_dtype_fp8():
    m = make_modeler(kv_data_type=DataType.FP8)
    assert m._calc_k_size() == 256 * 128 * 1  # 32768

def test_v_size_uses_kv_dtype():
    """V size = s2_base * d_base * kv_elem_size (same as K)"""
    m_fp16 = make_modeler(kv_data_type=DataType.FP16)
    m_fp8 = make_modeler(kv_data_type=DataType.FP8)
    assert m_fp16._calc_v_size() == 256 * 128 * 2
    assert m_fp8._calc_v_size() == 256 * 128 * 1

def test_simulation_produces_fixpipe_events():
    """Simulation still runs correctly after size changes"""
    m = make_modeler(s1_total=128, s2_total=256, d_total=128,
                     s1_base_size=128, s2_base_size=256, d_base_size=128)
    timeline, _, _, total_cycles = m.run_simulation()
    fixpipe_events = [e for e in timeline if e.unit == "FIXPIPE"]
    assert len(fixpipe_events) > 0
    assert total_cycles > 0

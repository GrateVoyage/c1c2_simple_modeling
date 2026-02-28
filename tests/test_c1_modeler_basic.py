# tests/test_c1_modeler_basic.py
import pytest
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(**kwargs):
    defaults = dict(
        s1_total=256, s2_total=512, d_total=128,
        s1_base_size=128, s2_base_size=256, d_base_size=128,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=128, baseN_C1=256, baseK_C1=128,
        baseM_C2=128, baseN_C2=128, baseK_C2=256,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
    )
    defaults.update(kwargs)
    return C1Modeler(**defaults)

def test_instantiation_new_api():
    m = make_modeler()
    assert m.q_data_type == DataType.FP16
    assert m.kv_data_type == DataType.FP16
    assert m.inter_core_pipeline == InterCorePipeline.DEFAULT
    assert m.inner_core_pipeline == InnerCorePipeline.DEFAULT

def test_instantiation_fp8():
    m = make_modeler(q_data_type=DataType.FP8, kv_data_type=DataType.FP8)
    assert m.q_data_type == DataType.FP8
    assert m.kv_data_type == DataType.FP8

def test_q_element_size_fp16():
    m = make_modeler(q_data_type=DataType.FP16)
    assert m._get_q_element_size() == 2

def test_q_element_size_fp8():
    m = make_modeler(q_data_type=DataType.FP8)
    assert m._get_q_element_size() == 1

def test_kv_element_size_fp16():
    m = make_modeler(kv_data_type=DataType.FP16)
    assert m._get_kv_element_size() == 2

def test_kv_element_size_fp8():
    m = make_modeler(kv_data_type=DataType.FP8)
    assert m._get_kv_element_size() == 1

def test_base_params_stored():
    m = make_modeler(baseM_C1=64, baseN_C1=64, baseK_C1=64,
                     baseM_C2=32, baseN_C2=32, baseK_C2=32)
    assert m.baseM_C1 == 64
    assert m.baseN_C1 == 64
    assert m.baseK_C1 == 64
    assert m.baseM_C2 == 32
    assert m.baseN_C2 == 32
    assert m.baseK_C2 == 32

def test_simulation_runs_default():
    """DEFAULT pipeline runs without error"""
    m = make_modeler(s1_total=128, s2_total=256, d_total=128,
                     s1_base_size=128, s2_base_size=256, d_base_size=128)
    timeline, bound_type, unit_times, total_cycles = m.run_simulation()
    assert total_cycles > 0
    assert len(timeline) > 0

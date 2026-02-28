import pytest
from core import InterCorePipeline, InnerCorePipeline, BoundType, DataType, LoadOrder

def test_inter_core_pipeline_values():
    assert InterCorePipeline.DEFAULT.value == "default"
    assert InterCorePipeline.PRELOAD.value == "preload"
    assert InterCorePipeline.N_BUFFER.value == "n_buffer"

def test_inner_core_pipeline_values():
    assert InnerCorePipeline.DEFAULT.value == "default"

def test_all_inter_core_pipeline_members():
    members = {e.value for e in InterCorePipeline}
    assert members == {"default", "preload", "n_buffer"}

def test_all_inner_core_pipeline_members():
    members = {e.value for e in InnerCorePipeline}
    assert members == {"default"}

def test_bound_type_members():
    members = {e.value for e in BoundType}
    assert members == {"MTE2_BOUND", "MTE1_BOUND", "MAC_BOUND", "FIXPIPE_BOUND"}

def test_data_type_members():
    members = {e.value for e in DataType}
    assert members == {"fp16", "fp8"}

def test_load_order_members():
    assert LoadOrder.LOAD_Q_FIRST.value == 0
    assert LoadOrder.LOAD_K_FIRST.value == 1

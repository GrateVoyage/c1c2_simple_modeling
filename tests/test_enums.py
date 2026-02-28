import pytest
from core import InterCorePipeline, InnerCorePipeline

def test_inter_core_pipeline_values():
    assert InterCorePipeline.DEFAULT.value == "default"
    assert InterCorePipeline.PRELOAD.value == "preload"
    assert InterCorePipeline.N_BUFFER.value == "n_buffer"

def test_inner_core_pipeline_values():
    assert InnerCorePipeline.DEFAULT.value == "default"

def test_all_inter_core_pipeline_members():
    members = {e.value for e in InterCorePipeline}
    assert members == {"default", "preload", "n_buffer"}

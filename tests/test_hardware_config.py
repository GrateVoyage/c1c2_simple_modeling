# tests/test_hardware_config.py
from core import HardwareConfig, DataType

hw = HardwareConfig()

def test_mte1_bytes_per_cycle():
    """MTE1 带宽 256 bytes/cycle"""
    assert hw.MTE1_BYTES_PER_CYCLE == 256.0

def test_fixpipe_bytes_per_cycle():
    """FIXPIPE 带宽 128 bytes/cycle"""
    assert hw.FIXPIPE_BYTES_PER_CYCLE == 128.0

def test_mte3_bytes_per_cycle():
    """MTE3 带宽 256 bytes/cycle"""
    assert hw.MTE3_BYTES_PER_CYCLE == 256.0

def test_mac_throughput_c1_both_fp16():
    """两者都是FP16 → FP16吞吐"""
    assert hw.get_mac_throughput_c1(DataType.FP16, DataType.FP16) == hw.MAC_THROUGHPUT_FP16

def test_mac_throughput_c1_both_fp8():
    """两者都是FP8 → FP8吞吐"""
    assert hw.get_mac_throughput_c1(DataType.FP8, DataType.FP8) == hw.MAC_THROUGHPUT_FP8

def test_mac_throughput_c1_mixed_fp16_fp8():
    """Q=FP16, KV=FP8 → 仍为FP16吞吐"""
    assert hw.get_mac_throughput_c1(DataType.FP16, DataType.FP8) == hw.MAC_THROUGHPUT_FP16

def test_mac_throughput_c1_mixed_fp8_fp16():
    """Q=FP8, KV=FP16 → 仍为FP16吞吐"""
    assert hw.get_mac_throughput_c1(DataType.FP8, DataType.FP16) == hw.MAC_THROUGHPUT_FP16

def test_mac_throughput_c2_fp16():
    assert hw.get_mac_throughput_c2(DataType.FP16) == hw.MAC_THROUGHPUT_FP16

def test_mac_throughput_c2_fp8():
    assert hw.get_mac_throughput_c2(DataType.FP8) == hw.MAC_THROUGHPUT_FP8

# tests/test_matmul_split.py
import pytest
from modelers import C1Modeler
from core import DataType, InterCorePipeline, InnerCorePipeline

def make_modeler(s1_base=128, s2_base=256, d_base=128,
                 baseN_C1=256, baseK_C1=128,
                 baseN_C2=128, baseK_C2=256,
                 **kwargs):
    defaults = dict(
        s1_total=128, s2_total=s2_base, d_total=d_base,
        s1_base_size=s1_base, s2_base_size=s2_base, d_base_size=d_base,
        q_data_type=DataType.FP16, kv_data_type=DataType.FP16,
        baseM_C1=s1_base, baseN_C1=baseN_C1, baseK_C1=baseK_C1,
        baseM_C2=s1_base, baseN_C2=baseN_C2, baseK_C2=baseK_C2,
        inter_core_pipeline=InterCorePipeline.DEFAULT,
        inner_core_pipeline=InnerCorePipeline.DEFAULT,
        L1_db=False, L0_db=False,
    )
    defaults.update(kwargs)
    return C1Modeler(**defaults)

def test_matmul_full_single_mac_c1():
    """matmulFull: d_base <= baseK_C1 AND s2_base <= baseN_C1 → single MAC P event"""
    # d_base=128 <= baseK_C1=128, s2_base=256 <= baseN_C1=256 → matmulFull
    m = make_modeler(s2_base=256, d_base=128, baseN_C1=256, baseK_C1=128)
    timeline, _, _, _ = m.run_simulation()
    mac_c1 = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    # 1 q_block × 1 k_block × 1 sub-MAC = 1 event
    assert len(mac_c1) == 1, f"Expected 1 MAC P event, got {len(mac_c1)}"

def test_matmulK_c1_two_sub_macs():
    """matmulK: d_base=256 > baseK_C1=128 → 2 sub-MACs, 1 FIXPIPE"""
    m = make_modeler(s2_base=256, d_base=256, baseN_C1=256, baseK_C1=128)
    timeline, _, _, _ = m.run_simulation()
    mac_c1 = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    fixpipe_p = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "P"]
    # 1 q_block × 1 k_block × 2 sub-MACs = 2 MAC P events
    assert len(mac_c1) == 2, f"Expected 2 MAC P events, got {len(mac_c1)}"
    # Only 1 FIXPIPE P (accumulate, then drain)
    assert len(fixpipe_p) == 1, f"Expected 1 FIXPIPE P, got {len(fixpipe_p)}"

def test_matmulK_c1_fixpipe_after_all_macs():
    """matmulK: FIXPIPE P start >= last MAC P end"""
    m = make_modeler(s2_base=256, d_base=256, baseN_C1=256, baseK_C1=128)
    timeline, _, _, _ = m.run_simulation()
    mac_c1 = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    fixpipe_p = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "P"]
    last_mac_end = max(e.end_time for e in mac_c1)
    assert fixpipe_p[0].start_time >= last_mac_end - 0.001, (
        f"FIXPIPE P should start after last MAC: "
        f"fixpipe_start={fixpipe_p[0].start_time:.1f}, last_mac_end={last_mac_end:.1f}"
    )

def test_matmulK_c2_two_sub_macs():
    """matmulK for C2: s2_base=256 > baseK_C2=128 → 2 sub-MACs, 1 FIXPIPE O"""
    # C2: K=s2_base=256 > baseK_C2=128 → split
    m = make_modeler(s2_base=256, d_base=128,
                     baseN_C1=256, baseK_C1=128,
                     baseN_C2=128, baseK_C2=128)
    timeline, _, _, _ = m.run_simulation()
    mac_c2 = [e for e in timeline if e.unit == "MAC" and e.operation == "O"]
    fixpipe_o = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "O"]
    assert len(mac_c2) == 2, f"Expected 2 MAC O events, got {len(mac_c2)}"
    assert len(fixpipe_o) == 1, f"Expected 1 FIXPIPE O, got {len(fixpipe_o)}"

def test_matmulK_c1_three_sub_macs():
    """matmulK: d_base=384 > baseK_C1=128 → ceil(384/128)=3 sub-MACs"""
    m = make_modeler(s2_base=256, d_base=384, baseN_C1=256, baseK_C1=128,
                     d_total=384)
    timeline, _, _, _ = m.run_simulation()
    mac_c1 = [e for e in timeline if e.unit == "MAC" and e.operation == "P"]
    fixpipe_p = [e for e in timeline if e.unit == "FIXPIPE" and e.operation == "P"]
    assert len(mac_c1) == 3, f"Expected 3 MAC P events, got {len(mac_c1)}"
    assert len(fixpipe_p) == 1, f"Expected 1 FIXPIPE P, got {len(fixpipe_p)}"

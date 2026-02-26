"""
Matmul建模器 - 纯数学流水线建模版

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# --- 枚举定义 ---
class BoundType(Enum):
    MTE2_BOUND = "MTE2_BOUND"
    MTE1_BOUND = "MTE1_BOUND"
    MAC_BOUND = "MAC_BOUND"
    FIXPIPE_BOUND = "FIXPIPE_BOUND"

class DataType(Enum):
    FP16 = "fp16"
    FP8 = "fp8"

class LoadOrder(Enum):
    LOAD_Q_FIRST = 0
    LOAD_K_FIRST = 1

# --- 数据结构 ---
@dataclass
class TimelineEvent:
    unit: str
    operation: str
    start_time: float
    end_time: float
    duration: float
    buffer: Optional[str] = None
    q_block_idx: int = 0
    k_block_idx: int = 0
    is_l2_hit: bool = False

# --- 模拟器核心 ---
class MatmulModeler:
    def __init__(
        self,
        s1_total: int = 256,
        s2_total: int = 1024,
        d_total: int = 128,
        s1_base_size: int = 128,
        s2_base_size: int = 256,
        d_base_size: int = 128,
        data_type: DataType = DataType.FP16,
        is_l2cache: bool = False,
        use_dn: bool = False,
        L1_db: bool = False,
        L0_db: bool = False,
        load_order: LoadOrder = LoadOrder.LOAD_Q_FIRST,
        full_load: bool = False,
    ):
        self.s1_total = s1_total
        self.s2_total = s2_total
        self.d_total = d_total
        self.s1_base = s1_base_size
        self.s2_base = s2_base_size
        self.d_base = d_base_size

        self.q_block_count = self.s1_total // self.s1_base
        self.k_block_count = self.s2_total // self.s2_base
        self.data_type = data_type
        self.is_l2cache = is_l2cache
        
        self.use_dn = use_dn
        self.L1_db = L1_db
        self.L0_db = L0_db
        self.load_order = load_order
        self.full_load = full_load
        
        self.timeline: List[TimelineEvent] = []

        # --- 硬件参数 ---
        self.CHIP_FREQ_GHZ = 1.65
        self.MTE2_DRAM_BW = 1600 * 1024 * 1024 * 1024 / 32
        self.MTE2_DRAM_BYTES_PER_CYCLE = self.MTE2_DRAM_BW / (self.CHIP_FREQ_GHZ * 1e9)
        self.MTE2_L2_BW = 5400 * 1024 * 1024 * 1024 / 32
        self.MTE2_L2_BYTES_PER_CYCLE = self.MTE2_L2_BW / (self.CHIP_FREQ_GHZ * 1e9)
        self.MTE1_FIXPIPE_BYTES_PER_CYCLE = 256.0

    def _get_element_size(self) -> int:
        return 2 if self.data_type == DataType.FP16 else 1

    # --- 计算公式 ---
    def _calc_mte2_cycles(self, size_bytes: int, use_l2: bool = False) -> float:
        bytes_per_cycle = self.MTE2_L2_BYTES_PER_CYCLE if use_l2 else self.MTE2_DRAM_BYTES_PER_CYCLE
        return size_bytes / bytes_per_cycle

    def _calc_mte1_cycles(self, size_bytes: int) -> float:
        return size_bytes / self.MTE1_FIXPIPE_BYTES_PER_CYCLE

    def _calc_mac_cycles(self, m, n, k) -> float:
        ops = m * n * k * 2
        throughput = 16 * 16 * 16 * 2 if self.data_type == DataType.FP16 else 16 * 32 * 16 * 2
        return ops / throughput

    def _calc_fixpipe_cycles(self, size_bytes: int) -> float:
        return size_bytes / self.MTE1_FIXPIPE_BYTES_PER_CYCLE

    def run_simulation(self):
        print(f"=== 开始数学流水线模拟 ===")
        print(f"特性: DN={'开启' if self.use_dn else '关闭'}, L1_DB={'开启' if self.L1_db else '关闭'}, L0_DB={'开启' if self.L0_db else '关闭'}")

        self.timeline.clear()
        resource_free_time = {"MTE2": 0.0, "MTE1": 0.0, "MAC": 0.0, "FIXPIPE": 0.0}
        
        # --- 缓冲区管理初始化 ---
        # L1 Buffer: L1A 和 L1B 独立管理
        num_l1_slots = 2 if self.L1_db else 1
        l1a_free_times = [0.0] * num_l1_slots
        l1b_free_times = [0.0] * num_l1_slots
        
        # L0 Buffer: L0A 和 L0B 独立管理 (修正点1)
        num_l0_slots = 2 if self.L0_db else 1
        l0a_free_times = [0.0] * num_l0_slots
        l0b_free_times = [0.0] * num_l0_slots
        
        # 状态记录
        q_l0_ready_times = {}
        q_l1_ready_times = {}

        # --- 确定 Q 和 K 的路径映射 (修正点2) ---
        # use_dn=False: Q走A路径(L1A->L0A), K走B路径(L1B->L0B)
        # use_dn=True:  Q走B路径(L1B->L0B), K走A路径(L1A->L0A)
        
        if self.use_dn:
            # DN模式: L1A/L0A 存储 K; L1B/L0B 存储 Q
            map_q = {'l1': 'L1B', 'l0': 'L0B', 'l1_slots': l1b_free_times, 'l0_slots': l0b_free_times}
            map_k = {'l1': 'L1A', 'l0': 'L0A', 'l1_slots': l1a_free_times, 'l0_slots': l0a_free_times}
        else:
            # 标准模式: L1A/L0A 存储 Q; L1B/L0B 存储 K
            map_q = {'l1': 'L1A', 'l0': 'L0A', 'l1_slots': l1a_free_times, 'l0_slots': l0a_free_times}
            map_k = {'l1': 'L1B', 'l0': 'L0B', 'l1_slots': l1b_free_times, 'l0_slots': l0b_free_times}

        # --- Preload Phase (Full Load) ---
        # Full Load 只针对 Q 矩阵
        if self.full_load:
            size_q = self.s1_base * self.d_base * self._get_element_size()
            dur_q = self._calc_mte2_cycles(size_q, use_l2=False)
            
            # Q 矩阵走 map_q 路径
            l1_name_q = map_q['l1']
            l1_slots_q = map_q['l1_slots']
            
            for q_idx in range(self.q_block_count):
                slot_idx = q_idx % len(l1_slots_q)
                
                start = max(resource_free_time["MTE2"], l1_slots_q[slot_idx])
                end = start + dur_q
                
                self.timeline.append(TimelineEvent("MTE2", f"Load {l1_name_q} (Q{q_idx+1})", start, end, dur_q, l1_name_q, q_idx, -1))
                resource_free_time["MTE2"] = end
                
                # Full Load 暂时占用 L1，假设 L0 搬运在后续循环中发生，这里先不释放 L1 Slot
                # 但为了模拟真实场景，Full Load 后 L1 其实是满的，直到搬运到 L0
                # 这里记录 L1 就绪时间
                q_l1_ready_times[q_idx] = end
                # 更新 L1 Slot 占用时间 (虽然 Full Load 通常意味着 L1 足够大，这里按逻辑更新)
                l1_slots_q[slot_idx] = end


        # === 外层循环：Q块 ===
        for q_idx in range(self.q_block_count):
            
            # --- 1. Q 矩阵搬运 ---
            # 路径: map_q
            
            # 1.1 L1 加载
            if not self.full_load:
                size_q = self.s1_base * self.d_base * self._get_element_size()
                dur_q_l1 = self._calc_mte2_cycles(size_q, use_l2=False)
                
                l1_name_q = map_q['l1']
                l1_slots_q = map_q['l1_slots']
                slot_l1_q = q_idx % len(l1_slots_q)
                
                start_l1_q = max(resource_free_time["MTE2"], l1_slots_q[slot_l1_q])
                end_l1_q = start_l1_q + dur_q_l1
                
                self.timeline.append(TimelineEvent("MTE2", f"Load {l1_name_q} (Q{q_idx+1})", start_l1_q, end_l1_q, dur_q_l1, l1_name_q, q_idx, -1))
                resource_free_time["MTE2"] = end_l1_q
                q_l1_ready_times[q_idx] = end_l1_q

            # 1.2 L0 搬运 (MTE1)
            size_q_l0 = self.s1_base * self.d_base * self._get_element_size()
            dur_q_l0 = self._calc_mte1_cycles(size_q_l0)
            
            l0_name_q = map_q['l0']
            l0_slots_q = map_q['l0_slots']
            l1_slots_q = map_q['l1_slots']
            
            slot_l0_q = q_idx % len(l0_slots_q)
            slot_l1_q = q_idx % len(l1_slots_q)
            
            start_l0_q = max(resource_free_time["MTE1"], q_l1_ready_times[q_idx], l0_slots_q[slot_l0_q])
            end_l0_q = start_l0_q + dur_q_l0
            
            self.timeline.append(TimelineEvent("MTE1", f"Load {l0_name_q} (Q{q_idx+1})", start_l0_q, end_l0_q, dur_q_l0, l0_name_q, q_idx, -1))
            resource_free_time["MTE1"] = end_l0_q
            q_l0_ready_times[q_idx] = end_l0_q
            
            # 释放 L1 Slot (数据已搬运至 L0)
            l1_slots_q[slot_l1_q] = end_l0_q
            
            # 注意：Q 是常量，在 L0 常驻。L0 Slot 的释放逻辑对于常量来说不是必须的，
            # 除非要覆盖。这里我们只对 K (流式数据) 应用严格的 L0 释放。

            # --- 2. 内层循环：K块 ---
            size_k_l1 = self.s2_base * self.d_base * self._get_element_size()
            size_k_l0 = size_k_l1
            
            # 路径: map_k
            l1_name_k = map_k['l1']
            l0_name_k = map_k['l0']
            l1_slots_k = map_k['l1_slots']
            l0_slots_k = map_k['l0_slots']

            for k_idx in range(self.k_block_count):
                # --- A. L1 加载 (MTE2) ---
                use_l2_for_k = self.is_l2cache and (q_idx > 0)
                dur_k_l1 = self._calc_mte2_cycles(size_k_l1, use_l2=use_l2_for_k)
                
                slot_l1_k = k_idx % len(l1_slots_k)
                
                start_l1_k = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
                end_l1_k = start_l1_k + dur_k_l1
                
                self.timeline.append(TimelineEvent("MTE2", f"Load {l1_name_k} (K{k_idx+1})", start_l1_k, end_l1_k, dur_k_l1, l1_name_k, q_idx, k_idx, is_l2_hit=use_l2_for_k))
                resource_free_time["MTE2"] = end_l1_k

                # --- B. L0 搬运 (MTE1) ---
                dur_k_l0 = self._calc_mte1_cycles(size_k_l0)
                
                slot_l0_k = k_idx % len(l0_slots_k)
                
                start_l0_k = max(resource_free_time["MTE1"], end_l1_k, l0_slots_k[slot_l0_k])
                end_l0_k = start_l0_k + dur_k_l0
                
                self.timeline.append(TimelineEvent("MTE1", f"Load {l0_name_k} (K{k_idx+1})", start_l0_k, end_l0_k, dur_k_l0, l0_name_k, q_idx, k_idx))
                resource_free_time["MTE1"] = end_l0_k
                
                # 释放 L1 Slot (数据已搬运至 L0)
                l1_slots_k[slot_l1_k] = end_l0_k

                # --- C. MAC 计算 ---
                dur_mac = self._calc_mac_cycles(self.s1_base, self.s2_base, self.d_base)
                start_mac = max(resource_free_time["MAC"], q_l0_ready_times[q_idx], end_l0_k)
                end_mac = start_mac + dur_mac
                self.timeline.append(TimelineEvent("MAC", "MAC", start_mac, end_mac, dur_mac, "L0C", q_idx, k_idx))
                resource_free_time["MAC"] = end_mac
                
                # --- 关键：释放 L0 Slot ---
                # K 是流式数据，计算完成后释放 L0 Buffer
                l0_slots_k[slot_l0_k] = end_mac

                # --- D. FIXPIPE ---
                size_out = self.s1_base * self.s2_base * self._get_element_size()
                dur_fix = self._calc_fixpipe_cycles(size_out)
                start_fix = max(resource_free_time["FIXPIPE"], end_mac)
                end_fix = start_fix + dur_fix
                self.timeline.append(TimelineEvent("FIXPIPE", "Out", start_fix, end_fix, dur_fix, "UB", q_idx, k_idx))
                resource_free_time["FIXPIPE"] = end_fix

        # 统计
        unit_times = {}
        total_cycles = 0.0
        if self.timeline:
            total_cycles = max(e.end_time for e in self.timeline)
            for e in self.timeline:
                unit_times[e.unit] = unit_times.get(e.unit, 0.0) + e.duration

        bound_type = max(
            [(bt, unit_times.get(bt.name.split('_')[0], 0)) for bt in BoundType],
            key=lambda x: x[1]
        )[0]

        print(f"=== 模拟结束 ===")
        return self.timeline, bound_type, unit_times, total_cycles

    # --- 绘图与输出 ---
    def plot_timeline(self, timeline: List[TimelineEvent], unit_times: Dict, total_cycles: float, filename: str = "timeline.png"):
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            fig, ax = plt.subplots(figsize=(14, 8))
            colors = {
                "MTE2": "#FF6B6B", "MTE2_L2": "#FFA07A",
                "MTE1": "#4ECDC4", "MAC": "#45B7D1", "FIXPIPE": "#96CEB4"
            }
            
            # 修正点3：Y轴标签简化，固定为 L0A/L0B/L1A/L1B
            y_map = {"FIXPIPE": 0, "MAC": 0.5, "L0B": 1, "L0A": 1.5, "L1B": 2, "L1A": 2.5}
            y_labels = ["FIXPIPE", "MAC", "L0B", "L0A", "L1B", "L1A"]

            for event in timeline:
                if event.unit not in colors: continue
                
                # 确定Y轴位置
                if event.unit == "MTE2":
                    y = y_map.get(event.buffer, 2.0) # L1A or L1B
                    color_key = "MTE2_L2" if event.is_l2_hit else "MTE2"
                elif event.unit == "MTE1":
                    y = y_map.get(event.buffer, 1.0) # L0A or L0B
                    color_key = "MTE1"
                else:
                    y = y_map[event.unit]
                    color_key = event.unit

                ax.barh(y, event.duration, left=event.start_time, height=0.3,
                        color=colors[color_key], edgecolor='black', alpha=0.8)

                # 标注文字
                label = ""
                if "Q" in event.operation: label = f"Q{event.q_block_idx+1}"
                elif "K" in event.operation: label = f"K{event.k_block_idx+1}"
                else: label = f"Q{event.q_block_idx+1}K{event.k_block_idx+1}"
                
                if event.is_l2_hit: label += "(L2)"

                if event.duration > total_cycles * 0.015:
                    ax.text(event.start_time + event.duration/2, y - 0.2, label,
                            ha='center', va='top', fontsize=8, color='black')

            ax.set_yticks(list(y_map.values()))
            ax.set_yticklabels(y_labels)
            ax.set_xlabel('Cycles')
            bound_name = max(unit_times, key=unit_times.get) if unit_times else "N/A"
            ax.set_title(f'Pipeline Timeline (Bound: {bound_name}, DN={self.use_dn}, L0_DB={self.L0_db})')
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)
            
            handles = [
                mpatches.Patch(color=colors["MTE2"], label='MTE2 (DRAM)'),
                mpatches.Patch(color=colors["MTE2_L2"], label='MTE2 (L2 Cache)'),
                mpatches.Patch(color=colors["MTE1"], label='MTE1'),
                mpatches.Patch(color=colors["MAC"], label='MAC'),
                mpatches.Patch(color=colors["FIXPIPE"], label='FIXPIPE')
            ]
            ax.legend(handles=handles, loc='upper right')
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            print(f"图表已保存: {filename}")
            plt.close()
        except Exception as e:
            logger.error(f"绘图错误: {e}", exc_info=True)

    def print_performance(self, unit_times: Dict, total_cycles: float):
        print("\n=== 性能分析 ===")
        print(f"总周期数: {total_cycles:.1f}")
        for unit in ["MTE2", "MTE1", "MAC", "FIXPIPE"]:
            t = unit_times.get(unit, 0)
            util = (t / total_cycles * 100) if total_cycles > 0 else 0
            print(f"{unit:<10} 总耗时: {t:<8.1f} (利用率: {util:.1f}%)")
        bound_name = max(unit_times, key=unit_times.get) if unit_times else "N/A"
        print(f"瓶颈分析: {bound_name}")

def main():
    # 测试场景：DN模式 + L0 DB
    # 预期结果：
    # 1. L1A 和 L0A 应该显示 K 的加载条（红色/青色条，标号为K1, K2...）
    # 2. L1B 和 L0B 应该显示 Q 的加载条（标号为Q1, Q2...）
    # 3. L0A 的 K 块之间会有交替（如果 L0_DB=True）
    modeler = MatmulModeler(
        s1_total=512, s2_total=2048, d_total=256,
        s1_base_size=128, s2_base_size=256, d_base_size=256,
        data_type=DataType.FP16,
        is_l2cache=False,
        use_dn=False,    # 开启 DN
        L1_db=True,
        L0_db=True,     # 开启 L0 DB
        full_load=False
    )

    timeline, bound_type, unit_times, total_cycles = modeler.run_simulation()
    modeler.print_performance(unit_times, total_cycles)
    modeler.plot_timeline(timeline, unit_times, total_cycles)

if __name__ == "__main__":
    main()
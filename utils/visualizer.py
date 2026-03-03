"""
时间线可视化工具
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免 WSL 环境下 Qt/Wayland 报错
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict
from core.dataclasses import TimelineEvent
from utils.logger import logger

class TimelineVisualizer:
    """时间线可视化器"""

    # 颜色配置
    COLORS = {
        "MTE2": "#FF6B6B",
        "MTE2_L2": "#FFA07A",
        "MTE1": "#4ECDC4",
        "MTE3": "#95A5A6",
        "MAC": "#45B7D1",
        "FIXPIPE": "#96CEB4",
        "VECTOR_V1": "#F7DC6F",
        "VECTOR_V2": "#E59866"
    }

    # Y轴映射 (从下到上: VECTOR_V2, VECTOR_V1, MTE3, FIXPIPE, MAC, MTE1B, MTE1A, MTE2B, MTE2A)
    Y_MAP = {
        "VECTOR_V2": 0,
        "VECTOR_V1": 0.3,
        "MTE3": 0.8,
        "FIXPIPE": 1.3,
        "MAC": 1.8,
        "MTE1B": 2.3,
        "MTE1A": 2.6,
        "MTE2B": 3.1,
        "MTE2A": 3.4
    }

    Y_LABELS = ["VECTOR_V2", "VECTOR_V1", "MTE3", "FIXPIPE", "MAC", "MTE1\n(L0B)", "MTE1\n(L0A)", "MTE2", "MTE2"]

    @staticmethod
    def plot_timeline(
        timeline: List[TimelineEvent],
        unit_times: Dict,
        total_cycles: float,
        use_dn: bool = False,
        L0_db: bool = False,
        filename: str = "timeline.png"
    ):
        """绘制时间线图表"""
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            fig, ax = plt.subplots(figsize=(14, 8))

            for event in timeline:
                if event.unit not in TimelineVisualizer.COLORS:
                    continue

                # 确定Y轴位置
                if event.unit == "MTE2":
                    y = TimelineVisualizer.Y_MAP.get(event.buffer, 3.5)
                    color_key = "MTE2_L2" if event.is_l2_hit else "MTE2"
                elif event.unit == "MTE1":
                    y = TimelineVisualizer.Y_MAP.get(event.buffer, 2.5)
                    color_key = "MTE1"
                elif event.unit == "MTE3":
                    y = TimelineVisualizer.Y_MAP["MTE3"]
                    color_key = "MTE3"
                elif event.unit in ["VECTOR_V1", "VECTOR_V2"]:
                    y = TimelineVisualizer.Y_MAP[event.unit]
                    color_key = event.unit
                else:
                    y = TimelineVisualizer.Y_MAP[event.unit]
                    color_key = event.unit

                ax.barh(y, event.duration, left=event.start_time, height=0.3,
                       color=TimelineVisualizer.COLORS[color_key],
                       edgecolor='black', alpha=0.8)

                # 标注文字
                label = ""
                if "Q" in event.operation:
                    label = f"Q{event.q_block_idx+1}"
                elif "K" in event.operation:
                    label = f"K{event.k_block_idx+1}"
                elif "V" in event.operation:
                    label = f"V{event.k_block_idx+1}"
                elif "P" in event.operation:
                    label = f"P{event.q_block_idx+1}{event.k_block_idx+1}"
                elif "O" in event.operation:
                    label = f"O{event.q_block_idx+1}{event.k_block_idx+1}"

                if event.is_l2_hit:
                    label += "(L2)"

                if event.duration > total_cycles * 0.003:
                    ax.text(event.start_time + event.duration/2, y - 0.2, label,
                           ha='center', va='top', fontsize=8, color='black')

            ax.set_yticks(list(TimelineVisualizer.Y_MAP.values()))
            ax.set_yticklabels(TimelineVisualizer.Y_LABELS)
            ax.set_xlabel('Cycles')

            bound_name = max(unit_times, key=unit_times.get) if unit_times else "N/A"
            ax.set_title(f'Pipeline Timeline (Bound: {bound_name}, DN={use_dn}, L0_DB={L0_db})')
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)

            handles = [
                mpatches.Patch(color=TimelineVisualizer.COLORS["MTE2"], label='MTE2 (DRAM)'),
                mpatches.Patch(color=TimelineVisualizer.COLORS["MTE2_L2"], label='MTE2 (L2 Cache)'),
                mpatches.Patch(color=TimelineVisualizer.COLORS["MTE1"], label='MTE1'),
                mpatches.Patch(color=TimelineVisualizer.COLORS["MTE3"], label='MTE3 (UB->L1)'),
                mpatches.Patch(color=TimelineVisualizer.COLORS["MAC"], label='MAC'),
                mpatches.Patch(color=TimelineVisualizer.COLORS["FIXPIPE"], label='FIXPIPE'),
                mpatches.Patch(color=TimelineVisualizer.COLORS["VECTOR_V1"], label='VECTOR V1'),
                mpatches.Patch(color=TimelineVisualizer.COLORS["VECTOR_V2"], label='VECTOR V2')
            ]
            ax.legend(handles=handles, loc='upper right')

            plt.tight_layout()

            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(filename, dpi=150)
            print(f"图表已保存: {filename}")
            plt.close()

        except Exception as e:
            logger.error(f"绘图错误: {e}", exc_info=True)

    @staticmethod
    def print_performance(unit_times: Dict, total_cycles: float):
        """打印性能分析"""
        print("\n=== 性能分析 ===")
        print(f"总周期数: {total_cycles:.1f}")

        for unit in ["MTE2", "MTE1", "MTE3", "MAC", "FIXPIPE", "VECTOR_V1", "VECTOR_V2"]:
            t = unit_times.get(unit, 0)
            util = (t / total_cycles * 100) if total_cycles > 0 else 0
            print(f"{unit:<10} 总耗时: {t:<8.1f} (利用率: {util:.1f}%)")

        bound_name = max(unit_times, key=unit_times.get) if unit_times else "N/A"
        print(f"瓶颈分析: {bound_name}")

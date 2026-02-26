"""
C1建模器 - 单矩阵乘法流水线建模
"""
from typing import List, Dict, Tuple
from core import BoundType, DataType, LoadOrder, TimelineEvent, HardwareConfig
from utils.visualizer import TimelineVisualizer

class C1Modeler:
    """C1建模器 - Flash Attention单矩阵乘法流水线"""

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
        preload: int = 0,
        hw_config: HardwareConfig = None,
    ):
        """
        初始化C1建模器

        Args:
            s1_total: Q矩阵的sequence长度
            s2_total: K矩阵的sequence长度
            d_total: 特征维度
            s1_base_size: Q块大小
            s2_base_size: K块大小
            d_base_size: D块大小
            data_type: 数据类型
            is_l2cache: 是否使用L2缓存
            use_dn: 是否使用DN模式
            L1_db: L1是否使用双缓冲
            L0_db: L0是否使用双缓冲
            load_order: 加载顺序
            full_load: 是否全量加载Q矩阵
            preload: 预加载模式 (0=正常, 1=先执行2次C1再进行C2)
            hw_config: 硬件配置
        """
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
        self.preload = preload

        self.timeline: List[TimelineEvent] = []

        # 硬件配置
        self.hw_config = hw_config if hw_config else HardwareConfig()

        # 提取硬件参数
        self.CHIP_FREQ_GHZ = self.hw_config.CHIP_FREQ_GHZ
        self.MTE2_DRAM_BYTES_PER_CYCLE = self.hw_config.MTE2_DRAM_BYTES_PER_CYCLE
        self.MTE2_L2_BYTES_PER_CYCLE = self.hw_config.MTE2_L2_BYTES_PER_CYCLE
        self.MTE1_FIXPIPE_BYTES_PER_CYCLE = self.hw_config.MTE1_FIXPIPE_BYTES_PER_CYCLE

    def _get_element_size(self) -> int:
        """获取数据元素大小"""
        return 2 if self.data_type == DataType.FP16 else 1

    def _calc_mte2_cycles(self, size_bytes: int, use_l2: bool = False) -> float:
        """计算MTE2搬运周期数"""
        bytes_per_cycle = self.MTE2_L2_BYTES_PER_CYCLE if use_l2 else self.MTE2_DRAM_BYTES_PER_CYCLE
        return size_bytes / bytes_per_cycle

    def _calc_mte1_cycles(self, size_bytes: int) -> float:
        """计算MTE1搬运周期数"""
        return size_bytes / self.MTE1_FIXPIPE_BYTES_PER_CYCLE

    def _calc_mac_cycles(self, m: int, n: int, k: int) -> float:
        """计算MAC计算周期数"""
        ops = m * n * k * 2
        throughput = self.hw_config.get_mac_throughput(self.data_type)
        return ops / throughput

    def _calc_fixpipe_cycles(self, size_bytes: int) -> float:
        """计算FIXPIPE处理周期数"""
        return size_bytes / self.MTE1_FIXPIPE_BYTES_PER_CYCLE

    def _calc_mte3_cycles(self, size_bytes: int) -> float:
        """计算MTE3搬运周期数 (UB->CUBE)"""
        # MTE3: 256 bytes/cycle
        return size_bytes / 256.0

    def _calc_vector_v1_cycles(self) -> float:
        """计算Vector V1处理周期数 (Softmax等操作)"""
        return 1600.0

    def _calc_vector_v2_cycles(self) -> float:
        """计算Vector V2处理周期数 (后处理操作)"""
        return 400.0

    def run_simulation(self) -> Tuple[List[TimelineEvent], BoundType, Dict, float]:
        """
        运行流水线模拟

        Returns:
            (timeline, bound_type, unit_times, total_cycles)
        """
        print(f"=== 开始C1流水线模拟 ===")
        print(f"特性: DN={'开启' if self.use_dn else '关闭'}, "
              f"L1_DB={'开启' if self.L1_db else '关闭'}, "
              f"L0_DB={'开启' if self.L0_db else '关闭'}")

        self.timeline.clear()
        resource_free_time = {"MTE2": 0.0, "MTE1": 0.0, "MAC": 0.0, "FIXPIPE": 0.0, "MTE3": 0.0, "VECTOR_V1": 0.0, "VECTOR_V2": 0.0}

        # 缓冲区管理初始化
        num_l1_slots = 2 if self.L1_db else 1
        l1a_free_times = [0.0] * num_l1_slots
        l1b_free_times = [0.0] * num_l1_slots

        num_l0_slots = 2 if self.L0_db else 1
        l0a_free_times = [0.0] * num_l0_slots
        l0b_free_times = [0.0] * num_l0_slots

        # 状态记录
        q_l0_ready_times = {}
        q_l1_ready_times = {}

        # 确定Q和K的路径映射
        if self.use_dn:
            # DN模式: L1A/L0A存储K; L1B/L0B存储Q
            map_q = {'l1': 'L1B', 'l0': 'L0B', 'l1_slots': l1b_free_times, 'l0_slots': l0b_free_times}
            map_k = {'l1': 'L1A', 'l0': 'L0A', 'l1_slots': l1a_free_times, 'l0_slots': l0a_free_times}
        else:
            # 标准模式: L1A/L0A存储Q; L1B/L0B存储K
            map_q = {'l1': 'L1A', 'l0': 'L0A', 'l1_slots': l1a_free_times, 'l0_slots': l0a_free_times}
            map_k = {'l1': 'L1B', 'l0': 'L0B', 'l1_slots': l1b_free_times, 'l0_slots': l0b_free_times}

        # Preload Phase (Full Load)
        if self.full_load:
            self._preload_q_blocks(map_q, resource_free_time, q_l1_ready_times)

        # 主循环：根据preload模式选择不同的执行策略
        if self.preload == 1:
            # Preload模式：先执行所有C1阶段，再执行所有C2阶段
            print("Preload模式: 先执行2次C1，再执行C2")

            # 记录P矩阵的就绪时间，用于C2阶段
            p_ready_times = {}

            # 阶段1: 执行所有C1+V1
            for q_idx in range(self.q_block_count):
                self._load_q_block(
                    q_idx, map_q, resource_free_time,
                    q_l1_ready_times, q_l0_ready_times
                )

                for k_idx in range(self.k_block_count):
                    p_ready_time = self._process_c1_stage(
                        q_idx, k_idx, map_k, resource_free_time,
                        q_l0_ready_times
                    )
                    p_ready_times[(q_idx, k_idx)] = p_ready_time

            # 阶段2: 执行所有C2+V2
            for q_idx in range(self.q_block_count):
                for k_idx in range(self.k_block_count):
                    self._process_c2_stage(
                        q_idx, k_idx, map_k, resource_free_time,
                        p_ready_times[(q_idx, k_idx)]
                    )
        else:
            # 正常模式：C1V1C2V2流水线连续执行
            for q_idx in range(self.q_block_count):
                self._load_q_block(
                    q_idx, map_q, resource_free_time,
                    q_l1_ready_times, q_l0_ready_times
                )

                self._process_k_blocks(
                    q_idx, map_k, resource_free_time,
                    q_l0_ready_times
                )

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

    def _preload_q_blocks(self, map_q: Dict, resource_free_time: Dict, q_l1_ready_times: Dict):
        """预加载Q块 (Full Load)"""
        size_q = self.s1_base * self.d_base * self._get_element_size()
        dur_q = self._calc_mte2_cycles(size_q, use_l2=False)

        l1_name_q = map_q['l1']
        l1_slots_q = map_q['l1_slots']

        for q_idx in range(self.q_block_count):
            slot_idx = q_idx % len(l1_slots_q)

            start = max(resource_free_time["MTE2"], l1_slots_q[slot_idx])
            end = start + dur_q

            self.timeline.append(TimelineEvent(
                "MTE2", f"Load {l1_name_q} (Q{q_idx+1})",
                start, end, dur_q, l1_name_q, q_idx, -1
            ))
            resource_free_time["MTE2"] = end
            q_l1_ready_times[q_idx] = end
            l1_slots_q[slot_idx] = end

    def _load_q_block(
        self, q_idx: int, map_q: Dict, resource_free_time: Dict,
        q_l1_ready_times: Dict, q_l0_ready_times: Dict
    ):
        """加载Q块"""
        # L1加载
        if not self.full_load:
            size_q = self.s1_base * self.d_base * self._get_element_size()
            dur_q_l1 = self._calc_mte2_cycles(size_q, use_l2=False)

            l1_name_q = map_q['l1']
            l1_slots_q = map_q['l1_slots']
            slot_l1_q = q_idx % len(l1_slots_q)

            start_l1_q = max(resource_free_time["MTE2"], l1_slots_q[slot_l1_q])
            end_l1_q = start_l1_q + dur_q_l1

            self.timeline.append(TimelineEvent(
                "MTE2", f"Load {l1_name_q} (Q{q_idx+1})",
                start_l1_q, end_l1_q, dur_q_l1, l1_name_q, q_idx, -1
            ))
            resource_free_time["MTE2"] = end_l1_q
            q_l1_ready_times[q_idx] = end_l1_q

        # L0搬运
        size_q_l0 = self.s1_base * self.d_base * self._get_element_size()
        dur_q_l0 = self._calc_mte1_cycles(size_q_l0)

        l0_name_q = map_q['l0']
        l0_slots_q = map_q['l0_slots']
        l1_slots_q = map_q['l1_slots']

        slot_l0_q = q_idx % len(l0_slots_q)
        slot_l1_q = q_idx % len(l1_slots_q)

        start_l0_q = max(
            resource_free_time["MTE1"],
            q_l1_ready_times[q_idx],
            l0_slots_q[slot_l0_q]
        )
        end_l0_q = start_l0_q + dur_q_l0

        self.timeline.append(TimelineEvent(
            "MTE1", f"Load {l0_name_q} (Q{q_idx+1})",
            start_l0_q, end_l0_q, dur_q_l0, l0_name_q, q_idx, -1
        ))
        resource_free_time["MTE1"] = end_l0_q
        q_l0_ready_times[q_idx] = end_l0_q
        l1_slots_q[slot_l1_q] = end_l0_q

    def _process_c1_stage(
        self, q_idx: int, k_idx: int, map_k: Dict, resource_free_time: Dict,
        q_l0_ready_times: Dict
    ) -> float:
        """执行C1+V1阶段，返回P矩阵的就绪时间"""
        size_k_l1 = self.s2_base * self.d_base * self._get_element_size()
        size_k_l0 = size_k_l1

        l1_name_k = map_k['l1']
        l0_name_k = map_k['l0']
        l1_slots_k = map_k['l1_slots']
        l0_slots_k = map_k['l0_slots']

        # A. L1加载 (MTE2)
        use_l2_for_k = self.is_l2cache and (q_idx > 0)
        dur_k_l1 = self._calc_mte2_cycles(size_k_l1, use_l2=use_l2_for_k)

        slot_l1_k = k_idx % len(l1_slots_k)

        start_l1_k = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
        end_l1_k = start_l1_k + dur_k_l1

        self.timeline.append(TimelineEvent(
            "MTE2", f"Load {l1_name_k} (K{k_idx+1})",
            start_l1_k, end_l1_k, dur_k_l1, l1_name_k,
            q_idx, k_idx, is_l2_hit=use_l2_for_k
        ))
        resource_free_time["MTE2"] = end_l1_k

        # B. L0搬运 (MTE1)
        dur_k_l0 = self._calc_mte1_cycles(size_k_l0)

        slot_l0_k = k_idx % len(l0_slots_k)

        start_l0_k = max(
            resource_free_time["MTE1"],
            end_l1_k,
            l0_slots_k[slot_l0_k]
        )
        end_l0_k = start_l0_k + dur_k_l0

        self.timeline.append(TimelineEvent(
            "MTE1", f"Load {l0_name_k} (K{k_idx+1})",
            start_l0_k, end_l0_k, dur_k_l0, l0_name_k, q_idx, k_idx
        ))
        resource_free_time["MTE1"] = end_l0_k
        l1_slots_k[slot_l1_k] = end_l0_k

        # C. MAC计算 (C1)
        dur_mac_c1 = self._calc_mac_cycles(self.s1_base, self.s2_base, self.d_base)
        start_mac_c1 = max(
            resource_free_time["MAC"],
            q_l0_ready_times[q_idx],
            end_l0_k
        )
        end_mac_c1 = start_mac_c1 + dur_mac_c1

        self.timeline.append(TimelineEvent(
            "MAC", "P", start_mac_c1, end_mac_c1, dur_mac_c1,
            "L0C", q_idx, k_idx
        ))
        resource_free_time["MAC"] = end_mac_c1
        l0_slots_k[slot_l0_k] = end_mac_c1

        # D. FIXPIPE (搬运P矩阵到UB)
        size_p = self.s1_base * self.s2_base * self._get_element_size()
        dur_fix_p = self._calc_fixpipe_cycles(size_p)
        start_fix_p = max(resource_free_time["FIXPIPE"], end_mac_c1)
        end_fix_p = start_fix_p + dur_fix_p

        self.timeline.append(TimelineEvent(
            "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
            "UB", q_idx, k_idx
        ))
        resource_free_time["FIXPIPE"] = end_fix_p

        # E. VECTOR_V1 (Softmax等操作)
        dur_vector_v1 = self._calc_vector_v1_cycles()
        start_vector_v1 = max(resource_free_time["VECTOR_V1"], end_fix_p)
        end_vector_v1 = start_vector_v1 + dur_vector_v1

        self.timeline.append(TimelineEvent(
            "VECTOR_V1", "P", start_vector_v1, end_vector_v1, dur_vector_v1,
            "CUBE", q_idx, k_idx
        ))
        resource_free_time["VECTOR_V1"] = end_vector_v1

        # 返回P矩阵就绪时间（用于C2阶段）
        return end_vector_v1

    def _process_c2_stage(
        self, q_idx: int, k_idx: int, map_k: Dict, resource_free_time: Dict,
        p_ready_time: float
    ):
        """执行C2+V2阶段"""
        size_p = self.s1_base * self.s2_base * self._get_element_size()

        l1_slots_v = map_k['l1_slots']
        l0_slots_v = map_k['l0_slots']

        # 路径映射
        if self.use_dn:
            l1_name_v = "L1A"
            l0_name_v = "L0A"
        else:
            l1_name_v = "L1B"
            l0_name_v = "L0B"

        # F. MTE3 (P从UB搬回CUBE)
        dur_mte3 = self._calc_mte3_cycles(size_p)
        start_mte3 = max(resource_free_time["MTE3"], p_ready_time)
        end_mte3 = start_mte3 + dur_mte3

        self.timeline.append(TimelineEvent(
            "MTE3", "P", start_mte3, end_mte3, dur_mte3,
            "CUBE", q_idx, k_idx
        ))
        resource_free_time["MTE3"] = end_mte3

        # G. 加载V矩阵到L1 (MTE2)
        size_v_l1 = self.s2_base * self.d_base * self._get_element_size()
        dur_v_l1 = self._calc_mte2_cycles(size_v_l1, use_l2=False)

        slot_l1_v = k_idx % len(l1_slots_v)
        start_l1_v = max(resource_free_time["MTE2"], l1_slots_v[slot_l1_v], p_ready_time)
        end_l1_v = start_l1_v + dur_v_l1

        self.timeline.append(TimelineEvent(
            "MTE2", f"Load {l1_name_v} (V{k_idx+1})",
            start_l1_v, end_l1_v, dur_v_l1, l1_name_v, q_idx, k_idx
        ))
        resource_free_time["MTE2"] = end_l1_v

        # H. V矩阵搬运到L0 (MTE1)
        size_v_l0 = size_v_l1
        dur_v_l0 = self._calc_mte1_cycles(size_v_l0)

        slot_l0_v = k_idx % len(l0_slots_v)
        start_l0_v = max(resource_free_time["MTE1"], end_l1_v, l0_slots_v[slot_l0_v])
        end_l0_v = start_l0_v + dur_v_l0

        self.timeline.append(TimelineEvent(
            "MTE1", f"Load {l0_name_v} (V{k_idx+1})",
            start_l0_v, end_l0_v, dur_v_l0, l0_name_v, q_idx, k_idx
        ))
        resource_free_time["MTE1"] = end_l0_v
        l1_slots_v[slot_l1_v] = end_l0_v

        # I. MAC计算 (C2: P @ V -> O)
        dur_mac_c2 = self._calc_mac_cycles(self.s1_base, self.d_base, self.s2_base)
        start_mac_c2 = max(
            resource_free_time["MAC"],
            end_mte3,
            end_l0_v
        )
        end_mac_c2 = start_mac_c2 + dur_mac_c2

        self.timeline.append(TimelineEvent(
            "MAC", "O", start_mac_c2, end_mac_c2, dur_mac_c2,
            "L0C", q_idx, k_idx
        ))
        resource_free_time["MAC"] = end_mac_c2
        l0_slots_v[slot_l0_v] = end_mac_c2

        # J. FIXPIPE (搬运O矩阵到UB)
        size_o = self.s1_base * self.d_base * self._get_element_size()
        dur_fix_o = self._calc_fixpipe_cycles(size_o)
        start_fix_o = max(resource_free_time["FIXPIPE"], end_mac_c2)
        end_fix_o = start_fix_o + dur_fix_o

        self.timeline.append(TimelineEvent(
            "FIXPIPE", "O", start_fix_o, end_fix_o, dur_fix_o,
            "UB", q_idx, k_idx
        ))
        resource_free_time["FIXPIPE"] = end_fix_o

        # K. VECTOR_V2 (后处理操作)
        dur_vector_v2 = self._calc_vector_v2_cycles()
        start_vector_v2 = max(resource_free_time["VECTOR_V2"], end_fix_o)
        end_vector_v2 = start_vector_v2 + dur_vector_v2

        self.timeline.append(TimelineEvent(
            "VECTOR_V2", "O", start_vector_v2, end_vector_v2, dur_vector_v2,
            "CUBE", q_idx, k_idx
        ))
        resource_free_time["VECTOR_V2"] = end_vector_v2

    def _process_k_blocks(
        self, q_idx: int, map_k: Dict, resource_free_time: Dict,
        q_l0_ready_times: Dict
    ):
        """处理K块并执行MAC计算"""
        size_k_l1 = self.s2_base * self.d_base * self._get_element_size()
        size_k_l0 = size_k_l1

        l1_name_k = map_k['l1']
        l0_name_k = map_k['l0']
        l1_slots_k = map_k['l1_slots']
        l0_slots_k = map_k['l0_slots']

        for k_idx in range(self.k_block_count):
            # A. L1加载 (MTE2)
            use_l2_for_k = self.is_l2cache and (q_idx > 0)
            dur_k_l1 = self._calc_mte2_cycles(size_k_l1, use_l2=use_l2_for_k)

            slot_l1_k = k_idx % len(l1_slots_k)

            start_l1_k = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
            end_l1_k = start_l1_k + dur_k_l1

            self.timeline.append(TimelineEvent(
                "MTE2", f"Load {l1_name_k} (K{k_idx+1})",
                start_l1_k, end_l1_k, dur_k_l1, l1_name_k,
                q_idx, k_idx, is_l2_hit=use_l2_for_k
            ))
            resource_free_time["MTE2"] = end_l1_k

            # B. L0搬运 (MTE1)
            dur_k_l0 = self._calc_mte1_cycles(size_k_l0)

            slot_l0_k = k_idx % len(l0_slots_k)

            start_l0_k = max(
                resource_free_time["MTE1"],
                end_l1_k,
                l0_slots_k[slot_l0_k]
            )
            end_l0_k = start_l0_k + dur_k_l0

            self.timeline.append(TimelineEvent(
                "MTE1", f"Load {l0_name_k} (K{k_idx+1})",
                start_l0_k, end_l0_k, dur_k_l0, l0_name_k, q_idx, k_idx
            ))
            resource_free_time["MTE1"] = end_l0_k
            l1_slots_k[slot_l1_k] = end_l0_k

            # === C1阶段: Q @ K^T -> P ===

            # C. MAC计算 (C1)
            dur_mac_c1 = self._calc_mac_cycles(self.s1_base, self.s2_base, self.d_base)
            start_mac_c1 = max(
                resource_free_time["MAC"],
                q_l0_ready_times[q_idx],
                end_l0_k
            )
            end_mac_c1 = start_mac_c1 + dur_mac_c1

            self.timeline.append(TimelineEvent(
                "MAC", "P", start_mac_c1, end_mac_c1, dur_mac_c1,
                "L0C", q_idx, k_idx
            ))
            resource_free_time["MAC"] = end_mac_c1
            l0_slots_k[slot_l0_k] = end_mac_c1

            # D. FIXPIPE (搬运P矩阵到UB)
            size_p = self.s1_base * self.s2_base * self._get_element_size()
            dur_fix_p = self._calc_fixpipe_cycles(size_p)
            start_fix_p = max(resource_free_time["FIXPIPE"], end_mac_c1)
            end_fix_p = start_fix_p + dur_fix_p

            self.timeline.append(TimelineEvent(
                "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                "UB", q_idx, k_idx
            ))
            resource_free_time["FIXPIPE"] = end_fix_p

            # === V1阶段: Softmax ===

            # E. VECTOR_V1 (Softmax等操作)
            dur_vector_v1 = self._calc_vector_v1_cycles()
            start_vector_v1 = max(resource_free_time["VECTOR_V1"], end_fix_p)
            end_vector_v1 = start_vector_v1 + dur_vector_v1

            self.timeline.append(TimelineEvent(
                "VECTOR_V1", "P", start_vector_v1, end_vector_v1, dur_vector_v1,
                "CUBE", q_idx, k_idx
            ))
            resource_free_time["VECTOR_V1"] = end_vector_v1

            # === 准备C2阶段 ===

            # F. MTE3 (P从UB搬回CUBE)
            dur_mte3 = self._calc_mte3_cycles(size_p)
            start_mte3 = max(resource_free_time["MTE3"], end_vector_v1)
            end_mte3 = start_mte3 + dur_mte3

            self.timeline.append(TimelineEvent(
                "MTE3", "P", start_mte3, end_mte3, dur_mte3,
                "CUBE", q_idx, k_idx
            ))
            resource_free_time["MTE3"] = end_mte3

            # G. 加载V矩阵到L1 (MTE2)
            # DN模式: V使用L1A路径；标准模式: V使用L1B路径
            if self.use_dn:
                l1_name_v = "L1A"
                l0_name_v = "L0A"
                l1_slots_v = map_k['l1_slots']  # 复用K的路径slots
                l0_slots_v = map_k['l0_slots']
            else:
                l1_name_v = "L1B"
                l0_name_v = "L0B"
                l1_slots_v = map_k['l1_slots']  # 复用K的路径slots
                l0_slots_v = map_k['l0_slots']

            size_v_l1 = self.s2_base * self.d_base * self._get_element_size()
            dur_v_l1 = self._calc_mte2_cycles(size_v_l1, use_l2=False)

            slot_l1_v = k_idx % len(l1_slots_v)
            start_l1_v = max(resource_free_time["MTE2"], l1_slots_v[slot_l1_v], end_vector_v1)
            end_l1_v = start_l1_v + dur_v_l1

            self.timeline.append(TimelineEvent(
                "MTE2", f"Load {l1_name_v} (V{k_idx+1})",
                start_l1_v, end_l1_v, dur_v_l1, l1_name_v, q_idx, k_idx
            ))
            resource_free_time["MTE2"] = end_l1_v

            # H. V矩阵搬运到L0 (MTE1)
            size_v_l0 = size_v_l1
            dur_v_l0 = self._calc_mte1_cycles(size_v_l0)

            slot_l0_v = k_idx % len(l0_slots_v)
            start_l0_v = max(resource_free_time["MTE1"], end_l1_v, l0_slots_v[slot_l0_v])
            end_l0_v = start_l0_v + dur_v_l0

            self.timeline.append(TimelineEvent(
                "MTE1", f"Load {l0_name_v} (V{k_idx+1})",
                start_l0_v, end_l0_v, dur_v_l0, l0_name_v, q_idx, k_idx
            ))
            resource_free_time["MTE1"] = end_l0_v
            l1_slots_v[slot_l1_v] = end_l0_v

            # === C2阶段: P @ V -> O ===

            # I. MAC计算 (C2: P @ V -> O)
            dur_mac_c2 = self._calc_mac_cycles(self.s1_base, self.d_base, self.s2_base)
            start_mac_c2 = max(
                resource_free_time["MAC"],
                end_mte3,  # P已搬回CUBE
                end_l0_v   # V已在L0
            )
            end_mac_c2 = start_mac_c2 + dur_mac_c2

            self.timeline.append(TimelineEvent(
                "MAC", "O", start_mac_c2, end_mac_c2, dur_mac_c2,
                "L0C", q_idx, k_idx
            ))
            resource_free_time["MAC"] = end_mac_c2
            l0_slots_v[slot_l0_v] = end_mac_c2

            # J. FIXPIPE (搬运O矩阵到UB)
            size_o = self.s1_base * self.d_base * self._get_element_size()
            dur_fix_o = self._calc_fixpipe_cycles(size_o)
            start_fix_o = max(resource_free_time["FIXPIPE"], end_mac_c2)
            end_fix_o = start_fix_o + dur_fix_o

            self.timeline.append(TimelineEvent(
                "FIXPIPE", "O", start_fix_o, end_fix_o, dur_fix_o,
                "UB", q_idx, k_idx
            ))
            resource_free_time["FIXPIPE"] = end_fix_o

            # === V2阶段: 后处理 ===

            # K. VECTOR_V2 (后处理操作)
            dur_vector_v2 = self._calc_vector_v2_cycles()
            start_vector_v2 = max(resource_free_time["VECTOR_V2"], end_fix_o)
            end_vector_v2 = start_vector_v2 + dur_vector_v2

            self.timeline.append(TimelineEvent(
                "VECTOR_V2", "O", start_vector_v2, end_vector_v2, dur_vector_v2,
                "CUBE", q_idx, k_idx
            ))
            resource_free_time["VECTOR_V2"] = end_vector_v2

    def plot_timeline(self, timeline: List[TimelineEvent], unit_times: Dict,
                     total_cycles: float, filename: str = "timeline.png"):
        """绘制时间线图表"""
        TimelineVisualizer.plot_timeline(
            timeline, unit_times, total_cycles,
            self.use_dn, self.L0_db, filename
        )

    def print_performance(self, unit_times: Dict, total_cycles: float):
        """打印性能分析"""
        TimelineVisualizer.print_performance(unit_times, total_cycles)

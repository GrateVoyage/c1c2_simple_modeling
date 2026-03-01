"""
C1建模器 - 单矩阵乘法流水线建模
"""
import math
from typing import List, Dict, Tuple
from core import BoundType, DataType, LoadOrder, InterCorePipeline, InnerCorePipeline, TimelineEvent, HardwareConfig
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
        q_data_type: DataType = DataType.FP16,
        kv_data_type: DataType = DataType.FP16,
        baseM_C1: int = 128,
        baseN_C1: int = 128,
        baseK_C1: int = 128,
        baseM_C2: int = 128,
        baseN_C2: int = 128,
        baseK_C2: int = 128,
        is_l2cache: bool = False,
        use_dn: bool = False,
        L1_db: bool = False,
        L0_db: bool = False,
        load_order: LoadOrder = LoadOrder.LOAD_Q_FIRST,
        full_load: bool = False,
        inter_core_pipeline: InterCorePipeline = InterCorePipeline.DEFAULT,
        inner_core_pipeline: InnerCorePipeline = InnerCorePipeline.DEFAULT,
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
            q_data_type: Q矩阵数据类型
            kv_data_type: K/V矩阵数据类型
            baseM_C1: C1阶段M维度基本块大小
            baseN_C1: C1阶段N维度基本块大小
            baseK_C1: C1阶段K维度基本块大小
            baseM_C2: C2阶段M维度基本块大小
            baseN_C2: C2阶段N维度基本块大小
            baseK_C2: C2阶段K维度基本块大小
            is_l2cache: 是否使用L2缓存
            use_dn: 是否使用DN模式
            L1_db: L1是否使用双缓冲
            L0_db: L0是否使用双缓冲
            load_order: 加载顺序
            full_load: 是否全量加载Q矩阵
            inter_core_pipeline: 核间流水线模式
            inner_core_pipeline: 核内流水线模式
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
        self.q_data_type = q_data_type
        self.kv_data_type = kv_data_type
        self.baseM_C1 = baseM_C1
        self.baseN_C1 = baseN_C1
        self.baseK_C1 = baseK_C1
        self.baseM_C2 = baseM_C2
        self.baseN_C2 = baseN_C2
        self.baseK_C2 = baseK_C2
        self.is_l2cache = is_l2cache

        self.use_dn = use_dn
        self.L1_db = L1_db
        self.L0_db = L0_db
        self.load_order = load_order
        self.full_load = full_load
        self.inter_core_pipeline = inter_core_pipeline
        self.inner_core_pipeline = inner_core_pipeline

        self.timeline: List[TimelineEvent] = []

        # 硬件配置
        self.hw_config = hw_config if hw_config else HardwareConfig()

        # 提取硬件参数
        self.CHIP_FREQ_GHZ = self.hw_config.CHIP_FREQ_GHZ
        self.MTE2_DRAM_BYTES_PER_CYCLE = self.hw_config.MTE2_DRAM_BYTES_PER_CYCLE
        self.MTE2_L2_BYTES_PER_CYCLE = self.hw_config.MTE2_L2_BYTES_PER_CYCLE
        self.MTE1_BYTES_PER_CYCLE = self.hw_config.MTE1_BYTES_PER_CYCLE
        self.FIXPIPE_BYTES_PER_CYCLE = self.hw_config.FIXPIPE_BYTES_PER_CYCLE
        self.MTE3_BYTES_PER_CYCLE = self.hw_config.MTE3_BYTES_PER_CYCLE

    def _get_q_element_size(self) -> int:
        """Q矩阵元素大小 (bytes)"""
        return 2 if self.q_data_type == DataType.FP16 else 1

    def _get_kv_element_size(self) -> int:
        """K/V矩阵及P矩阵元素大小 (bytes)"""
        return 2 if self.kv_data_type == DataType.FP16 else 1

    def _get_element_size(self) -> int:
        """获取数据元素大小 (deprecated: use _get_q_element_size or _get_kv_element_size)"""
        return self._get_kv_element_size()

    def _calc_q_size(self) -> int:
        """Q矩阵搬运大小 (bytes): s1_base × d_base × q_elem"""
        return self.s1_base * self.d_base * self._get_q_element_size()

    def _calc_k_size(self) -> int:
        """K矩阵搬运大小 (bytes): s2_base × d_base × kv_elem"""
        return self.s2_base * self.d_base * self._get_kv_element_size()

    def _calc_v_size(self) -> int:
        """V矩阵搬运大小 (bytes): s2_base × d_base × kv_elem"""
        return self.s2_base * self.d_base * self._get_kv_element_size()

    def _calc_fixpipe_p_size(self) -> int:
        """FIXPIPE P: L0C→UB, L0C输出恒为FP32"""
        return self.s1_base * self.s2_base * 4

    def _calc_fixpipe_o_size(self) -> int:
        """FIXPIPE O: L0C→UB, L0C输出恒为FP32"""
        return self.s1_base * self.d_base * 4

    def _calc_mte3_p_size(self) -> int:
        """MTE3 P: UB→CUBE, P经V1后为kv_data_type格式"""
        return self.s1_base * self.s2_base * self._get_kv_element_size()

    def _get_c1_split(self) -> tuple:
        """
        返回 C1 的切分类型和子块数。
        Priority: check N first, then K.
        Returns: ('full', 1) | ('N', sub_count) | ('K', sub_count)
        """
        if self.s2_base > self.baseN_C1:
            return 'N', math.ceil(self.s2_base / self.baseN_C1)
        if self.d_base > self.baseK_C1:
            return 'K', math.ceil(self.d_base / self.baseK_C1)
        return 'full', 1

    def _get_c2_split(self) -> tuple:
        """
        返回 C2 的切分类型和子块数。
        C2: P(s1_base, s2_base) @ V(s2_base, d_base), M=s1_base, N=d_base, K=s2_base
        Priority: check N first (d_base vs baseN_C2), then K (s2_base vs baseK_C2).
        Returns: ('full', 1) | ('N', sub_count) | ('K', sub_count)
        """
        if self.d_base > self.baseN_C2:
            return 'N', math.ceil(self.d_base / self.baseN_C2)
        if self.s2_base > self.baseK_C2:
            return 'K', math.ceil(self.s2_base / self.baseK_C2)
        return 'full', 1

    def _calc_mte2_cycles(self, size_bytes: int, use_l2: bool = False) -> float:
        """计算MTE2搬运周期数"""
        bytes_per_cycle = self.MTE2_L2_BYTES_PER_CYCLE if use_l2 else self.MTE2_DRAM_BYTES_PER_CYCLE
        return size_bytes / bytes_per_cycle

    def _calc_mte1_cycles(self, size_bytes: int) -> float:
        """计算MTE1搬运周期数"""
        return size_bytes / self.MTE1_BYTES_PER_CYCLE

    def _calc_mac_cycles(self, m: int, n: int, k: int) -> float:
        """计算MAC计算周期数"""
        ops = m * n * k * 2
        throughput = self.hw_config.get_mac_throughput(self.kv_data_type)
        return ops / throughput

    def _calc_mac_cycles_c1(self, m: int, n: int, k: int) -> float:
        """C1阶段MAC计算周期数 (Q@K^T): 吞吐量由q_data_type和kv_data_type共同决定"""
        ops = m * n * k * 2
        return ops / self.hw_config.get_mac_throughput_c1(self.q_data_type, self.kv_data_type)

    def _calc_mac_cycles_c2(self, m: int, n: int, k: int) -> float:
        """C2阶段MAC计算周期数 (P@V): 吞吐量由kv_data_type决定"""
        ops = m * n * k * 2
        return ops / self.hw_config.get_mac_throughput_c2(self.kv_data_type)

    def _calc_fixpipe_cycles(self, size_bytes: int) -> float:
        """计算FIXPIPE处理周期数"""
        return size_bytes / self.FIXPIPE_BYTES_PER_CYCLE

    def _calc_mte3_cycles(self, size_bytes: int) -> float:
        """计算MTE3搬运周期数 (UB->CUBE)"""
        return size_bytes / self.MTE3_BYTES_PER_CYCLE

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
              f"L0_DB={'开启' if self.L0_db else '关闭'}, "
              f"流水线={self.inter_core_pipeline.value}")

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

        # 主循环：根据inter_core_pipeline模式选择不同的执行策略
        if self.inter_core_pipeline == InterCorePipeline.PRELOAD:
            # PRELOAD 渐进式流水：提前发射 C1，V 在 C2 阶段加载
            # 序列: C1 -> C1V1C2 -> C1V1C2V2 -> ... -> V1C2V2 -> C2V2 -> V2
            print(f"Preload模式: 提前发射C1，V在C2阶段加载（与DEFAULT相同）")

            for q_idx in range(self.q_block_count):
                self._load_q_block(
                    q_idx, map_q, resource_free_time,
                    q_l1_ready_times, q_l0_ready_times
                )

                c1_fix_ends = []  # 保存每个 k_idx 的 end_fix_p

                for k_idx in range(self.k_block_count):
                    # 立即发射 C1[k]（只等 C1 相关硬件资源：MTE2/MTE1/MAC/FIXPIPE）
                    end_fix_p = self._process_c1_only(
                        q_idx, k_idx, map_k, resource_free_time, q_l0_ready_times
                    )
                    c1_fix_ends.append(end_fix_p)

                    # 发射上一个 k 的 trailing V1 + C2 + V2
                    if k_idx > 0:
                        prev = k_idx - 1
                        end_v1 = self._process_v1_only(
                            q_idx, prev, resource_free_time, c1_fix_ends[prev]
                        )
                        self._process_c2_stage(
                            q_idx, prev, map_k, resource_free_time, end_v1
                        )

                # 收尾：最后一个 k 的 V1 + C2 + V2
                if self.k_block_count > 0:
                    last = self.k_block_count - 1
                    end_v1 = self._process_v1_only(
                        q_idx, last, resource_free_time, c1_fix_ends[last]
                    )
                    self._process_c2_stage(
                        q_idx, last, map_k, resource_free_time, end_v1
                    )
        elif self.inter_core_pipeline == InterCorePipeline.N_BUFFER:
            # N_BUFFER模式（N=2）：将k-block分组，组内按阶段批次执行
            # Phase1: 组内所有k的C1 → Phase2: 组内所有k的V1
            # Phase3: 组内所有k的C2 → Phase4: 组内所有k的V2
            # 关键优化：V1[k0]可与MAC C1[k1]并行（不同硬件单元）
            print(f"N_BUFFER模式(N=2): C1C1→V1V1→C2C2→V2V2批次流水")
            n_size = 2
            for q_idx in range(self.q_block_count):
                self._load_q_block(q_idx, map_q, resource_free_time,
                                   q_l1_ready_times, q_l0_ready_times)
                for group_start in range(0, self.k_block_count, n_size):
                    group = list(range(group_start,
                                       min(group_start + n_size, self.k_block_count)))
                    self._process_n_buffer_group(
                        q_idx, group, map_k, resource_free_time, q_l0_ready_times
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
        size_q = self._calc_q_size()
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
            size_q = self._calc_q_size()
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
        size_q_l0 = self._calc_q_size()
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
        l1_name_k = map_k['l1']
        l0_name_k = map_k['l0']
        l1_slots_k = map_k['l1_slots']
        l0_slots_k = map_k['l0_slots']

        use_l2_for_k = self.is_l2cache and (q_idx > 0)
        slot_l1_k = k_idx % len(l1_slots_k)
        slot_l0_k = k_idx % len(l0_slots_k)

        # C. MAC计算 (C1) with optional matmul splitting
        split_type, sub_count = self._get_c1_split()

        if split_type == 'N':
            # matmulN: per-sub-tile K load (MTE2 K_sub + MTE1 K_sub) inside loop
            sub_n = self.baseN_C1
            last_mac_end_c1 = 0.0
            end_l0_k = 0.0
            end_fix_p = 0.0
            for i in range(sub_count):
                actual_n = min(sub_n, self.s2_base - i * sub_n)
                size_k_sub = actual_n * self.d_base * self._get_kv_element_size()

                # MTE2 K_sub[i]: load sub-tile to L1
                dur_k_sub_l1 = self._calc_mte2_cycles(size_k_sub, use_l2=use_l2_for_k)
                start_l1_k_sub = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
                end_l1_k_sub = start_l1_k_sub + dur_k_sub_l1
                self.timeline.append(TimelineEvent(
                    "MTE2", f"Load {l1_name_k} (K{k_idx+1}_sub{i})",
                    start_l1_k_sub, end_l1_k_sub, dur_k_sub_l1, l1_name_k,
                    q_idx, k_idx, is_l2_hit=use_l2_for_k
                ))
                resource_free_time["MTE2"] = end_l1_k_sub

                # MTE1 K_sub[i]: move sub-tile from L1 to L0B
                dur_k_sub_l0 = self._calc_mte1_cycles(size_k_sub)
                start_l0_k_sub = max(
                    resource_free_time["MTE1"],
                    end_l1_k_sub,
                    l0_slots_k[slot_l0_k]
                )
                end_l0_k_sub = start_l0_k_sub + dur_k_sub_l0
                self.timeline.append(TimelineEvent(
                    "MTE1", f"Load {l0_name_k} (K{k_idx+1}_sub{i})",
                    start_l0_k_sub, end_l0_k_sub, dur_k_sub_l0, l0_name_k, q_idx, k_idx
                ))
                resource_free_time["MTE1"] = end_l0_k_sub
                l1_slots_k[slot_l1_k] = end_l0_k_sub
                end_l0_k = end_l0_k_sub

                dur_mac_sub = self._calc_mac_cycles_c1(self.s1_base, actual_n, self.d_base)
                if i == 0:
                    start_mac_sub = max(resource_free_time["MAC"],
                                        q_l0_ready_times[q_idx], end_l0_k_sub)
                else:
                    # Next sub-MAC can start after previous FIXPIPE completes
                    start_mac_sub = max(resource_free_time["MAC"], end_fix_p, end_l0_k_sub)
                end_mac_sub = start_mac_sub + dur_mac_sub
                self.timeline.append(TimelineEvent(
                    "MAC", "P", start_mac_sub, end_mac_sub, dur_mac_sub,
                    "L0C", q_idx, k_idx
                ))
                resource_free_time["MAC"] = end_mac_sub
                last_mac_end_c1 = end_mac_sub

                # Each sub-tile gets its own FIXPIPE (no L0C accumulation)
                size_p_sub = self.s1_base * actual_n * 4  # FP32 output
                dur_fix_p = self._calc_fixpipe_cycles(size_p_sub)
                start_fix_p = max(resource_free_time["FIXPIPE"], end_mac_sub)
                end_fix_p = start_fix_p + dur_fix_p
                self.timeline.append(TimelineEvent(
                    "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                    "UB", q_idx, k_idx
                ))
                resource_free_time["FIXPIPE"] = end_fix_p

            l0_slots_k[slot_l0_k] = last_mac_end_c1

        else:
            # For 'K' and 'full': monolithic K load before MAC
            size_k = self._calc_k_size()

            # A. L1加载 (MTE2)
            dur_k_l1 = self._calc_mte2_cycles(size_k, use_l2=use_l2_for_k)
            start_l1_k = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
            end_l1_k = start_l1_k + dur_k_l1
            self.timeline.append(TimelineEvent(
                "MTE2", f"Load {l1_name_k} (K{k_idx+1})",
                start_l1_k, end_l1_k, dur_k_l1, l1_name_k,
                q_idx, k_idx, is_l2_hit=use_l2_for_k
            ))
            resource_free_time["MTE2"] = end_l1_k

            # B. L0搬运 (MTE1)
            dur_k_l0 = self._calc_mte1_cycles(size_k)
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

            if split_type == 'K':
                # matmulK: split K (d_base) dimension, L0C accumulates
                sub_k = math.ceil(self.d_base / sub_count)
                last_mac_end_c1 = end_l0_k
                for i in range(sub_count):
                    actual_k = min(sub_k, self.d_base - i * sub_k)
                    dur_mac_sub = self._calc_mac_cycles_c1(self.s1_base, self.s2_base, actual_k)
                    if i == 0:
                        start_mac_sub = max(resource_free_time["MAC"],
                                            q_l0_ready_times[q_idx], end_l0_k)
                    else:
                        start_mac_sub = resource_free_time["MAC"]
                    end_mac_sub = start_mac_sub + dur_mac_sub
                    self.timeline.append(TimelineEvent(
                        "MAC", "P", start_mac_sub, end_mac_sub, dur_mac_sub,
                        "L0C", q_idx, k_idx
                    ))
                    resource_free_time["MAC"] = end_mac_sub
                    last_mac_end_c1 = end_mac_sub

                # One FIXPIPE after all sub-MACs complete
                size_p = self._calc_fixpipe_p_size()
                dur_fix_p = self._calc_fixpipe_cycles(size_p)
                start_fix_p = max(resource_free_time["FIXPIPE"], last_mac_end_c1)
                end_fix_p = start_fix_p + dur_fix_p
                self.timeline.append(TimelineEvent(
                    "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                    "UB", q_idx, k_idx
                ))
                resource_free_time["FIXPIPE"] = end_fix_p
                l0_slots_k[slot_l0_k] = last_mac_end_c1

            else:
                # matmulFull
                dur_mac_c1 = self._calc_mac_cycles_c1(self.s1_base, self.s2_base, self.d_base)
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
                size_p = self._calc_fixpipe_p_size()
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

    def _process_c1_only(
        self, q_idx: int, k_idx: int, map_k: Dict, resource_free_time: Dict,
        q_l0_ready_times: Dict
    ) -> float:
        """执行C1阶段（K加载 + MAC-P + FIXPIPE-P），不包含VECTOR_V1，返回end_fix_p"""
        l1_name_k = map_k['l1']
        l0_name_k = map_k['l0']
        l1_slots_k = map_k['l1_slots']
        l0_slots_k = map_k['l0_slots']

        use_l2_for_k = self.is_l2cache and (q_idx > 0)
        slot_l1_k = k_idx % len(l1_slots_k)
        slot_l0_k = k_idx % len(l0_slots_k)

        split_type, sub_count = self._get_c1_split()

        if split_type == 'N':
            sub_n = self.baseN_C1
            last_mac_end_c1 = 0.0
            end_l0_k = 0.0
            end_fix_p = 0.0
            for i in range(sub_count):
                actual_n = min(sub_n, self.s2_base - i * sub_n)
                size_k_sub = actual_n * self.d_base * self._get_kv_element_size()

                dur_k_sub_l1 = self._calc_mte2_cycles(size_k_sub, use_l2=use_l2_for_k)
                start_l1_k_sub = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
                end_l1_k_sub = start_l1_k_sub + dur_k_sub_l1
                self.timeline.append(TimelineEvent(
                    "MTE2", f"Load {l1_name_k} (K{k_idx+1}_sub{i})",
                    start_l1_k_sub, end_l1_k_sub, dur_k_sub_l1, l1_name_k,
                    q_idx, k_idx, is_l2_hit=use_l2_for_k
                ))
                resource_free_time["MTE2"] = end_l1_k_sub

                dur_k_sub_l0 = self._calc_mte1_cycles(size_k_sub)
                start_l0_k_sub = max(
                    resource_free_time["MTE1"],
                    end_l1_k_sub,
                    l0_slots_k[slot_l0_k]
                )
                end_l0_k_sub = start_l0_k_sub + dur_k_sub_l0
                self.timeline.append(TimelineEvent(
                    "MTE1", f"Load {l0_name_k} (K{k_idx+1}_sub{i})",
                    start_l0_k_sub, end_l0_k_sub, dur_k_sub_l0, l0_name_k, q_idx, k_idx
                ))
                resource_free_time["MTE1"] = end_l0_k_sub
                l1_slots_k[slot_l1_k] = end_l0_k_sub
                end_l0_k = end_l0_k_sub

                dur_mac_sub = self._calc_mac_cycles_c1(self.s1_base, actual_n, self.d_base)
                if i == 0:
                    start_mac_sub = max(resource_free_time["MAC"],
                                        q_l0_ready_times[q_idx], end_l0_k_sub)
                else:
                    start_mac_sub = max(resource_free_time["MAC"], end_fix_p, end_l0_k_sub)
                end_mac_sub = start_mac_sub + dur_mac_sub
                self.timeline.append(TimelineEvent(
                    "MAC", "P", start_mac_sub, end_mac_sub, dur_mac_sub,
                    "L0C", q_idx, k_idx
                ))
                resource_free_time["MAC"] = end_mac_sub
                last_mac_end_c1 = end_mac_sub

                size_p_sub = self.s1_base * actual_n * 4
                dur_fix_p = self._calc_fixpipe_cycles(size_p_sub)
                start_fix_p = max(resource_free_time["FIXPIPE"], end_mac_sub)
                end_fix_p = start_fix_p + dur_fix_p
                self.timeline.append(TimelineEvent(
                    "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                    "UB", q_idx, k_idx
                ))
                resource_free_time["FIXPIPE"] = end_fix_p

            l0_slots_k[slot_l0_k] = last_mac_end_c1

        else:
            size_k = self._calc_k_size()

            dur_k_l1 = self._calc_mte2_cycles(size_k, use_l2=use_l2_for_k)
            start_l1_k = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
            end_l1_k = start_l1_k + dur_k_l1
            self.timeline.append(TimelineEvent(
                "MTE2", f"Load {l1_name_k} (K{k_idx+1})",
                start_l1_k, end_l1_k, dur_k_l1, l1_name_k,
                q_idx, k_idx, is_l2_hit=use_l2_for_k
            ))
            resource_free_time["MTE2"] = end_l1_k

            dur_k_l0 = self._calc_mte1_cycles(size_k)
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

            if split_type == 'K':
                sub_k = math.ceil(self.d_base / sub_count)
                last_mac_end_c1 = end_l0_k
                for i in range(sub_count):
                    actual_k = min(sub_k, self.d_base - i * sub_k)
                    dur_mac_sub = self._calc_mac_cycles_c1(self.s1_base, self.s2_base, actual_k)
                    if i == 0:
                        start_mac_sub = max(resource_free_time["MAC"],
                                            q_l0_ready_times[q_idx], end_l0_k)
                    else:
                        start_mac_sub = resource_free_time["MAC"]
                    end_mac_sub = start_mac_sub + dur_mac_sub
                    self.timeline.append(TimelineEvent(
                        "MAC", "P", start_mac_sub, end_mac_sub, dur_mac_sub,
                        "L0C", q_idx, k_idx
                    ))
                    resource_free_time["MAC"] = end_mac_sub
                    last_mac_end_c1 = end_mac_sub

                size_p = self._calc_fixpipe_p_size()
                dur_fix_p = self._calc_fixpipe_cycles(size_p)
                start_fix_p = max(resource_free_time["FIXPIPE"], last_mac_end_c1)
                end_fix_p = start_fix_p + dur_fix_p
                self.timeline.append(TimelineEvent(
                    "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                    "UB", q_idx, k_idx
                ))
                resource_free_time["FIXPIPE"] = end_fix_p
                l0_slots_k[slot_l0_k] = last_mac_end_c1

            else:
                dur_mac_c1 = self._calc_mac_cycles_c1(self.s1_base, self.s2_base, self.d_base)
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

                size_p = self._calc_fixpipe_p_size()
                dur_fix_p = self._calc_fixpipe_cycles(size_p)
                start_fix_p = max(resource_free_time["FIXPIPE"], end_mac_c1)
                end_fix_p = start_fix_p + dur_fix_p
                self.timeline.append(TimelineEvent(
                    "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                    "UB", q_idx, k_idx
                ))
                resource_free_time["FIXPIPE"] = end_fix_p

        return end_fix_p

    def _process_v1_only(
        self, q_idx: int, k_idx: int, resource_free_time: Dict, end_fix_p: float
    ) -> float:
        """发射VECTOR_V1事件，返回end_vector_v1"""
        dur = self._calc_vector_v1_cycles()
        start = max(resource_free_time["VECTOR_V1"], end_fix_p)
        end = start + dur
        self.timeline.append(TimelineEvent("VECTOR_V1", "P", start, end, dur, "CUBE", q_idx, k_idx))
        resource_free_time["VECTOR_V1"] = end
        return end

    def _process_c2_stage(
        self, q_idx: int, k_idx: int, map_k: Dict, resource_free_time: Dict,
        p_ready_time: float
    ):
        """执行C2+V2阶段"""
        size_p_mte3 = self._calc_mte3_p_size()

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
        dur_mte3 = self._calc_mte3_cycles(size_p_mte3)
        start_mte3 = max(resource_free_time["MTE3"], p_ready_time)
        end_mte3 = start_mte3 + dur_mte3

        self.timeline.append(TimelineEvent(
            "MTE3", "P", start_mte3, end_mte3, dur_mte3,
            "CUBE", q_idx, k_idx
        ))
        resource_free_time["MTE3"] = end_mte3

        # I. MAC计算 (C2: P @ V -> O) with optional matmul splitting
        split_type_c2, sub_count_c2 = self._get_c2_split()

        slot_l1_v = k_idx % len(l1_slots_v)
        slot_l0_v = k_idx % len(l0_slots_v)

        if split_type_c2 == 'N':
            # matmulN for C2: per-sub-tile V load (MTE2 V_sub + MTE1 V_sub) inside loop
            sub_n = self.baseN_C2
            last_mac_end_c2 = 0.0
            end_l0_v = 0.0
            end_fix_o = 0.0
            for i in range(sub_count_c2):
                actual_n = min(sub_n, self.d_base - i * sub_n)
                size_v_sub = self.s2_base * actual_n * self._get_kv_element_size()

                # MTE2 V_sub[i]: load sub-tile to L1
                dur_v_sub_l1 = self._calc_mte2_cycles(size_v_sub, use_l2=False)
                start_l1_v_sub = max(resource_free_time["MTE2"], l1_slots_v[slot_l1_v], p_ready_time)
                end_l1_v_sub = start_l1_v_sub + dur_v_sub_l1
                self.timeline.append(TimelineEvent(
                    "MTE2", f"Load {l1_name_v} (V{k_idx+1}_sub{i})",
                    start_l1_v_sub, end_l1_v_sub, dur_v_sub_l1, l1_name_v, q_idx, k_idx
                ))
                resource_free_time["MTE2"] = end_l1_v_sub

                # MTE1 V_sub[i]: move sub-tile from L1 to L0
                dur_v_sub_l0 = self._calc_mte1_cycles(size_v_sub)
                start_l0_v_sub = max(resource_free_time["MTE1"], end_l1_v_sub, l0_slots_v[slot_l0_v])
                end_l0_v_sub = start_l0_v_sub + dur_v_sub_l0
                self.timeline.append(TimelineEvent(
                    "MTE1", f"Load {l0_name_v} (V{k_idx+1}_sub{i})",
                    start_l0_v_sub, end_l0_v_sub, dur_v_sub_l0, l0_name_v, q_idx, k_idx
                ))
                resource_free_time["MTE1"] = end_l0_v_sub
                l1_slots_v[slot_l1_v] = end_l0_v_sub
                end_l0_v = end_l0_v_sub

                dur_mac_sub = self._calc_mac_cycles_c2(self.s1_base, actual_n, self.s2_base)
                if i == 0:
                    start_mac_sub = max(resource_free_time["MAC"], end_mte3, end_l0_v_sub)
                else:
                    # Next sub-MAC can start after previous FIXPIPE completes
                    start_mac_sub = max(resource_free_time["MAC"], end_fix_o, end_l0_v_sub)
                end_mac_sub = start_mac_sub + dur_mac_sub
                self.timeline.append(TimelineEvent(
                    "MAC", "O", start_mac_sub, end_mac_sub, dur_mac_sub,
                    "L0C", q_idx, k_idx
                ))
                resource_free_time["MAC"] = end_mac_sub
                last_mac_end_c2 = end_mac_sub

                # Each sub-tile gets its own FIXPIPE (no L0C accumulation)
                size_o_sub = self.s1_base * actual_n * 4  # FP32 output
                dur_fix_o = self._calc_fixpipe_cycles(size_o_sub)
                start_fix_o = max(resource_free_time["FIXPIPE"], end_mac_sub)
                end_fix_o = start_fix_o + dur_fix_o
                self.timeline.append(TimelineEvent(
                    "FIXPIPE", "O", start_fix_o, end_fix_o, dur_fix_o,
                    "UB", q_idx, k_idx
                ))
                resource_free_time["FIXPIPE"] = end_fix_o

            l0_slots_v[slot_l0_v] = last_mac_end_c2

        else:
            # For 'K' and 'full': monolithic V load before MAC
            size_v_l1 = self._calc_v_size()
            dur_v_l1 = self._calc_mte2_cycles(size_v_l1, use_l2=False)
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
            start_l0_v = max(resource_free_time["MTE1"], end_l1_v, l0_slots_v[slot_l0_v])
            end_l0_v = start_l0_v + dur_v_l0
            self.timeline.append(TimelineEvent(
                "MTE1", f"Load {l0_name_v} (V{k_idx+1})",
                start_l0_v, end_l0_v, dur_v_l0, l0_name_v, q_idx, k_idx
            ))
            resource_free_time["MTE1"] = end_l0_v
            l1_slots_v[slot_l1_v] = end_l0_v

            if split_type_c2 == 'K':
                # matmulK for C2: split s2_base dimension
                sub_k = math.ceil(self.s2_base / sub_count_c2)
                last_mac_end_c2 = 0.0
                for i in range(sub_count_c2):
                    actual_k = min(sub_k, self.s2_base - i * sub_k)
                    dur_mac_sub = self._calc_mac_cycles_c2(self.s1_base, self.d_base, actual_k)
                    if i == 0:
                        start_mac_sub = max(resource_free_time["MAC"], end_mte3, end_l0_v)
                    else:
                        start_mac_sub = resource_free_time["MAC"]
                    end_mac_sub = start_mac_sub + dur_mac_sub
                    self.timeline.append(TimelineEvent(
                        "MAC", "O", start_mac_sub, end_mac_sub, dur_mac_sub,
                        "L0C", q_idx, k_idx
                    ))
                    resource_free_time["MAC"] = end_mac_sub
                    last_mac_end_c2 = end_mac_sub

                size_o = self._calc_fixpipe_o_size()
                dur_fix_o = self._calc_fixpipe_cycles(size_o)
                start_fix_o = max(resource_free_time["FIXPIPE"], last_mac_end_c2)
                end_fix_o = start_fix_o + dur_fix_o
                self.timeline.append(TimelineEvent(
                    "FIXPIPE", "O", start_fix_o, end_fix_o, dur_fix_o,
                    "UB", q_idx, k_idx
                ))
                resource_free_time["FIXPIPE"] = end_fix_o
                l0_slots_v[slot_l0_v] = last_mac_end_c2

            else:
                # matmulFull
                dur_mac_c2 = self._calc_mac_cycles_c2(self.s1_base, self.d_base, self.s2_base)
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
                size_o = self._calc_fixpipe_o_size()
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
        l1_name_k = map_k['l1']
        l0_name_k = map_k['l0']
        l1_slots_k = map_k['l1_slots']
        l0_slots_k = map_k['l0_slots']

        for k_idx in range(self.k_block_count):
            use_l2_for_k = self.is_l2cache and (q_idx > 0)
            slot_l1_k = k_idx % len(l1_slots_k)
            slot_l0_k = k_idx % len(l0_slots_k)

            # === C1阶段: Q @ K^T -> P (with optional matmul splitting) ===
            split_type, sub_count = self._get_c1_split()

            if split_type == 'N':
                # matmulN: per-sub-tile K load (MTE2 K_sub + MTE1 K_sub) inside loop
                sub_n = self.baseN_C1
                last_mac_end_c1 = 0.0
                end_l0_k = 0.0
                end_fix_p = 0.0
                for i in range(sub_count):
                    actual_n = min(sub_n, self.s2_base - i * sub_n)
                    size_k_sub = actual_n * self.d_base * self._get_kv_element_size()

                    # MTE2 K_sub[i]: load sub-tile to L1
                    dur_k_sub_l1 = self._calc_mte2_cycles(size_k_sub, use_l2=use_l2_for_k)
                    start_l1_k_sub = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
                    end_l1_k_sub = start_l1_k_sub + dur_k_sub_l1
                    self.timeline.append(TimelineEvent(
                        "MTE2", f"Load {l1_name_k} (K{k_idx+1}_sub{i})",
                        start_l1_k_sub, end_l1_k_sub, dur_k_sub_l1, l1_name_k,
                        q_idx, k_idx, is_l2_hit=use_l2_for_k
                    ))
                    resource_free_time["MTE2"] = end_l1_k_sub

                    # MTE1 K_sub[i]: move sub-tile from L1 to L0B
                    dur_k_sub_l0 = self._calc_mte1_cycles(size_k_sub)
                    start_l0_k_sub = max(
                        resource_free_time["MTE1"],
                        end_l1_k_sub,
                        l0_slots_k[slot_l0_k]
                    )
                    end_l0_k_sub = start_l0_k_sub + dur_k_sub_l0
                    self.timeline.append(TimelineEvent(
                        "MTE1", f"Load {l0_name_k} (K{k_idx+1}_sub{i})",
                        start_l0_k_sub, end_l0_k_sub, dur_k_sub_l0, l0_name_k, q_idx, k_idx
                    ))
                    resource_free_time["MTE1"] = end_l0_k_sub
                    l1_slots_k[slot_l1_k] = end_l0_k_sub
                    end_l0_k = end_l0_k_sub

                    dur_mac_sub = self._calc_mac_cycles_c1(self.s1_base, actual_n, self.d_base)
                    if i == 0:
                        start_mac_sub = max(resource_free_time["MAC"],
                                            q_l0_ready_times[q_idx], end_l0_k_sub)
                    else:
                        # Next sub-MAC can start after previous FIXPIPE completes
                        start_mac_sub = max(resource_free_time["MAC"], end_fix_p, end_l0_k_sub)
                    end_mac_sub = start_mac_sub + dur_mac_sub
                    self.timeline.append(TimelineEvent(
                        "MAC", "P", start_mac_sub, end_mac_sub, dur_mac_sub,
                        "L0C", q_idx, k_idx
                    ))
                    resource_free_time["MAC"] = end_mac_sub
                    last_mac_end_c1 = end_mac_sub

                    # Each sub-tile gets its own FIXPIPE (no L0C accumulation)
                    size_p_sub = self.s1_base * actual_n * 4  # FP32 output
                    dur_fix_p = self._calc_fixpipe_cycles(size_p_sub)
                    start_fix_p = max(resource_free_time["FIXPIPE"], end_mac_sub)
                    end_fix_p = start_fix_p + dur_fix_p
                    self.timeline.append(TimelineEvent(
                        "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                        "UB", q_idx, k_idx
                    ))
                    resource_free_time["FIXPIPE"] = end_fix_p

                l0_slots_k[slot_l0_k] = last_mac_end_c1

            else:
                # For 'K' and 'full': monolithic K load before MAC
                size_k = self._calc_k_size()

                # A. L1加载 (MTE2)
                dur_k_l1 = self._calc_mte2_cycles(size_k, use_l2=use_l2_for_k)
                start_l1_k = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
                end_l1_k = start_l1_k + dur_k_l1
                self.timeline.append(TimelineEvent(
                    "MTE2", f"Load {l1_name_k} (K{k_idx+1})",
                    start_l1_k, end_l1_k, dur_k_l1, l1_name_k,
                    q_idx, k_idx, is_l2_hit=use_l2_for_k
                ))
                resource_free_time["MTE2"] = end_l1_k

                # B. L0搬运 (MTE1)
                dur_k_l0 = self._calc_mte1_cycles(size_k)
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

                if split_type == 'K':
                    # matmulK: split K (d_base) dimension, L0C accumulates
                    sub_k = math.ceil(self.d_base / sub_count)
                    last_mac_end_c1 = end_l0_k  # initial dependency
                    for i in range(sub_count):
                        actual_k = min(sub_k, self.d_base - i * sub_k)
                        dur_mac_sub = self._calc_mac_cycles_c1(self.s1_base, self.s2_base, actual_k)
                        if i == 0:
                            start_mac_sub = max(resource_free_time["MAC"],
                                                q_l0_ready_times[q_idx], end_l0_k)
                        else:
                            start_mac_sub = resource_free_time["MAC"]
                        end_mac_sub = start_mac_sub + dur_mac_sub
                        self.timeline.append(TimelineEvent(
                            "MAC", "P", start_mac_sub, end_mac_sub, dur_mac_sub,
                            "L0C", q_idx, k_idx
                        ))
                        resource_free_time["MAC"] = end_mac_sub
                        last_mac_end_c1 = end_mac_sub

                    # One FIXPIPE after all sub-MACs complete
                    size_p_fix = self._calc_fixpipe_p_size()
                    dur_fix_p = self._calc_fixpipe_cycles(size_p_fix)
                    start_fix_p = max(resource_free_time["FIXPIPE"], last_mac_end_c1)
                    end_fix_p = start_fix_p + dur_fix_p
                    self.timeline.append(TimelineEvent(
                        "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                        "UB", q_idx, k_idx
                    ))
                    resource_free_time["FIXPIPE"] = end_fix_p
                    l0_slots_k[slot_l0_k] = last_mac_end_c1  # release L0 slot after last MAC

                else:
                    # matmulFull
                    dur_mac_c1 = self._calc_mac_cycles_c1(self.s1_base, self.s2_base, self.d_base)
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

                    size_p_fix = self._calc_fixpipe_p_size()
                    dur_fix_p = self._calc_fixpipe_cycles(size_p_fix)
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
            dur_mte3 = self._calc_mte3_cycles(self._calc_mte3_p_size())
            start_mte3 = max(resource_free_time["MTE3"], end_vector_v1)
            end_mte3 = start_mte3 + dur_mte3

            self.timeline.append(TimelineEvent(
                "MTE3", "P", start_mte3, end_mte3, dur_mte3,
                "CUBE", q_idx, k_idx
            ))
            resource_free_time["MTE3"] = end_mte3

            # DN模式: V使用L1A路径；标准模式: V使用L1B路径
            if self.use_dn:
                l1_name_v = "L1A"
                l0_name_v = "L0A"
                l1_slots_v = map_k['l1_slots']
                l0_slots_v = map_k['l0_slots']
            else:
                l1_name_v = "L1B"
                l0_name_v = "L0B"
                l1_slots_v = map_k['l1_slots']
                l0_slots_v = map_k['l0_slots']

            slot_l1_v = k_idx % len(l1_slots_v)
            slot_l0_v = k_idx % len(l0_slots_v)

            # === C2阶段: P @ V -> O (with optional matmul splitting) ===
            split_type_c2, sub_count_c2 = self._get_c2_split()

            if split_type_c2 == 'N':
                # matmulN for C2: per-sub-tile V load (MTE2 V_sub + MTE1 V_sub) inside loop
                sub_n_c2 = self.baseN_C2
                last_mac_end_c2 = 0.0
                end_l0_v = 0.0
                end_fix_o = 0.0
                for i in range(sub_count_c2):
                    actual_n = min(sub_n_c2, self.d_base - i * sub_n_c2)
                    size_v_sub = self.s2_base * actual_n * self._get_kv_element_size()

                    # MTE2 V_sub[i]: load sub-tile to L1
                    dur_v_sub_l1 = self._calc_mte2_cycles(size_v_sub, use_l2=False)
                    start_l1_v_sub = max(resource_free_time["MTE2"], l1_slots_v[slot_l1_v], end_vector_v1)
                    end_l1_v_sub = start_l1_v_sub + dur_v_sub_l1
                    self.timeline.append(TimelineEvent(
                        "MTE2", f"Load {l1_name_v} (V{k_idx+1}_sub{i})",
                        start_l1_v_sub, end_l1_v_sub, dur_v_sub_l1, l1_name_v, q_idx, k_idx
                    ))
                    resource_free_time["MTE2"] = end_l1_v_sub

                    # MTE1 V_sub[i]: move sub-tile from L1 to L0
                    dur_v_sub_l0 = self._calc_mte1_cycles(size_v_sub)
                    start_l0_v_sub = max(resource_free_time["MTE1"], end_l1_v_sub, l0_slots_v[slot_l0_v])
                    end_l0_v_sub = start_l0_v_sub + dur_v_sub_l0
                    self.timeline.append(TimelineEvent(
                        "MTE1", f"Load {l0_name_v} (V{k_idx+1}_sub{i})",
                        start_l0_v_sub, end_l0_v_sub, dur_v_sub_l0, l0_name_v, q_idx, k_idx
                    ))
                    resource_free_time["MTE1"] = end_l0_v_sub
                    l1_slots_v[slot_l1_v] = end_l0_v_sub
                    end_l0_v = end_l0_v_sub

                    dur_mac_sub = self._calc_mac_cycles_c2(self.s1_base, actual_n, self.s2_base)
                    if i == 0:
                        start_mac_sub = max(resource_free_time["MAC"], end_mte3, end_l0_v_sub)
                    else:
                        # Next sub-MAC can start after previous FIXPIPE completes
                        start_mac_sub = max(resource_free_time["MAC"], end_fix_o, end_l0_v_sub)
                    end_mac_sub = start_mac_sub + dur_mac_sub
                    self.timeline.append(TimelineEvent(
                        "MAC", "O", start_mac_sub, end_mac_sub, dur_mac_sub,
                        "L0C", q_idx, k_idx
                    ))
                    resource_free_time["MAC"] = end_mac_sub
                    last_mac_end_c2 = end_mac_sub

                    # Each sub-tile gets its own FIXPIPE (no L0C accumulation)
                    size_fo_sub = self.s1_base * actual_n * 4  # FP32 output
                    dur_fix_o = self._calc_fixpipe_cycles(size_fo_sub)
                    start_fix_o = max(resource_free_time["FIXPIPE"], end_mac_sub)
                    end_fix_o = start_fix_o + dur_fix_o
                    self.timeline.append(TimelineEvent(
                        "FIXPIPE", "O", start_fix_o, end_fix_o, dur_fix_o,
                        "UB", q_idx, k_idx
                    ))
                    resource_free_time["FIXPIPE"] = end_fix_o

                l0_slots_v[slot_l0_v] = last_mac_end_c2

            else:
                # For 'K' and 'full': monolithic V load before MAC
                size_v_l1 = self._calc_v_size()
                dur_v_l1 = self._calc_mte2_cycles(size_v_l1, use_l2=False)
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
                start_l0_v = max(resource_free_time["MTE1"], end_l1_v, l0_slots_v[slot_l0_v])
                end_l0_v = start_l0_v + dur_v_l0
                self.timeline.append(TimelineEvent(
                    "MTE1", f"Load {l0_name_v} (V{k_idx+1})",
                    start_l0_v, end_l0_v, dur_v_l0, l0_name_v, q_idx, k_idx
                ))
                resource_free_time["MTE1"] = end_l0_v
                l1_slots_v[slot_l1_v] = end_l0_v

                if split_type_c2 == 'K':
                    # matmulK for C2: split s2_base dimension
                    sub_k = math.ceil(self.s2_base / sub_count_c2)
                    last_mac_end_c2 = 0.0
                    for i in range(sub_count_c2):
                        actual_k = min(sub_k, self.s2_base - i * sub_k)
                        dur_mac_sub = self._calc_mac_cycles_c2(self.s1_base, self.d_base, actual_k)
                        if i == 0:
                            start_mac_sub = max(resource_free_time["MAC"], end_mte3, end_l0_v)
                        else:
                            start_mac_sub = resource_free_time["MAC"]
                        end_mac_sub = start_mac_sub + dur_mac_sub
                        self.timeline.append(TimelineEvent(
                            "MAC", "O", start_mac_sub, end_mac_sub, dur_mac_sub,
                            "L0C", q_idx, k_idx
                        ))
                        resource_free_time["MAC"] = end_mac_sub
                        last_mac_end_c2 = end_mac_sub

                    size_fo = self._calc_fixpipe_o_size()
                    dur_fix_o = self._calc_fixpipe_cycles(size_fo)
                    start_fix_o = max(resource_free_time["FIXPIPE"], last_mac_end_c2)
                    end_fix_o = start_fix_o + dur_fix_o
                    self.timeline.append(TimelineEvent(
                        "FIXPIPE", "O", start_fix_o, end_fix_o, dur_fix_o,
                        "UB", q_idx, k_idx
                    ))
                    resource_free_time["FIXPIPE"] = end_fix_o
                    l0_slots_v[slot_l0_v] = last_mac_end_c2

                else:
                    # matmulFull
                    dur_mac_c2 = self._calc_mac_cycles_c2(self.s1_base, self.d_base, self.s2_base)
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

                    size_fo = self._calc_fixpipe_o_size()
                    dur_fix_o = self._calc_fixpipe_cycles(size_fo)
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

    def _process_n_buffer_group(
        self, q_idx: int, k_group: list, map_k: Dict,
        resource_free_time: Dict, q_l0_ready_times: Dict
    ):
        """
        N_BUFFER 组内按阶段批次执行：
        Phase1: 组内所有 k 的 C1（K-load + MTE1 + MAC + FIXPIPE）
        Phase2: 组内所有 k 的 V1（VECTOR_V1）— 可与 Phase1 末尾的 MAC 并行
        Phase3: 组内所有 k 的 C2（V-load + MTE3 + MTE1 + MAC C2 + FIXPIPE）
        Phase4: 组内所有 k 的 V2（VECTOR_V2）
        """
        l1_name_k = map_k['l1']
        l0_name_k = map_k['l0']
        l1_slots_k = map_k['l1_slots']
        l0_slots_k = map_k['l0_slots']

        # Determine V path
        if self.use_dn:
            l1_name_v, l0_name_v = "L1A", "L0A"
        else:
            l1_name_v, l0_name_v = "L1B", "L0B"
        l1_slots_v = map_k['l1_slots']
        l0_slots_v = map_k['l0_slots']

        fixpipe_p_ready = {}   # k_idx → FIXPIPE P end time
        v1_ready = {}          # k_idx → VECTOR_V1 end time
        fixpipe_o_ready = {}   # k_idx → FIXPIPE O end time

        # ── Phase 1: C1 for all k in group ──────────────────────────────
        for k_idx in k_group:
            use_l2_for_k = self.is_l2cache and (q_idx > 0)
            slot_l1_k = k_idx % len(l1_slots_k)
            slot_l0_k = k_idx % len(l0_slots_k)

            split_type_nb, sub_count_nb = self._get_c1_split()

            if split_type_nb == 'N':
                # matmulN: per-sub-tile K load (MTE2 K_sub + MTE1 K_sub) inside loop
                sub_n = self.baseN_C1
                last_mac_end_c1 = 0.0
                end_l0_k = 0.0
                for i in range(sub_count_nb):
                    actual_n = min(sub_n, self.s2_base - i * sub_n)
                    size_k_sub = actual_n * self.d_base * self._get_kv_element_size()

                    # MTE2 K_sub[i]: load sub-tile to L1
                    dur_k_sub_l1 = self._calc_mte2_cycles(size_k_sub, use_l2=use_l2_for_k)
                    start_l1_k_sub = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
                    end_l1_k_sub = start_l1_k_sub + dur_k_sub_l1
                    self.timeline.append(TimelineEvent(
                        "MTE2", f"Load {l1_name_k} (K{k_idx+1}_sub{i})",
                        start_l1_k_sub, end_l1_k_sub, dur_k_sub_l1, l1_name_k,
                        q_idx, k_idx, is_l2_hit=use_l2_for_k
                    ))
                    resource_free_time["MTE2"] = end_l1_k_sub

                    # MTE1 K_sub[i]: move sub-tile from L1 to L0B
                    dur_k_sub_l0 = self._calc_mte1_cycles(size_k_sub)
                    start_l0_k_sub = max(
                        resource_free_time["MTE1"],
                        end_l1_k_sub,
                        l0_slots_k[slot_l0_k]
                    )
                    end_l0_k_sub = start_l0_k_sub + dur_k_sub_l0
                    self.timeline.append(TimelineEvent(
                        "MTE1", f"Load {l0_name_k} (K{k_idx+1}_sub{i})",
                        start_l0_k_sub, end_l0_k_sub, dur_k_sub_l0, l0_name_k, q_idx, k_idx
                    ))
                    resource_free_time["MTE1"] = end_l0_k_sub
                    l1_slots_k[slot_l1_k] = end_l0_k_sub
                    end_l0_k = end_l0_k_sub

                    dur_mac_sub = self._calc_mac_cycles_c1(self.s1_base, actual_n, self.d_base)
                    if i == 0:
                        start_mac_sub = max(resource_free_time["MAC"],
                                            q_l0_ready_times[q_idx], end_l0_k_sub)
                    else:
                        # Next sub-MAC can start after previous FIXPIPE completes
                        start_mac_sub = max(resource_free_time["MAC"], end_fix_p, end_l0_k_sub)
                    end_mac_sub = start_mac_sub + dur_mac_sub
                    self.timeline.append(TimelineEvent(
                        "MAC", "P", start_mac_sub, end_mac_sub, dur_mac_sub,
                        "L0C", q_idx, k_idx
                    ))
                    resource_free_time["MAC"] = end_mac_sub
                    last_mac_end_c1 = end_mac_sub

                    # Each sub-tile gets its own FIXPIPE (no L0C accumulation)
                    size_fp_sub = self.s1_base * actual_n * 4  # FP32 output
                    dur_fix_p = self._calc_fixpipe_cycles(size_fp_sub)
                    start_fix_p = max(resource_free_time["FIXPIPE"], end_mac_sub)
                    end_fix_p = start_fix_p + dur_fix_p
                    self.timeline.append(TimelineEvent(
                        "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                        "UB", q_idx, k_idx
                    ))
                    resource_free_time["FIXPIPE"] = end_fix_p

                l0_slots_k[slot_l0_k] = last_mac_end_c1
                fixpipe_p_ready[k_idx] = end_fix_p

            else:
                # For 'K' and 'full': monolithic K load before MAC
                size_k = self._calc_k_size()
                dur_k_l1 = self._calc_mte2_cycles(size_k, use_l2=use_l2_for_k)
                start_l1_k = max(resource_free_time["MTE2"], l1_slots_k[slot_l1_k])
                end_l1_k = start_l1_k + dur_k_l1
                self.timeline.append(TimelineEvent(
                    "MTE2", f"Load {l1_name_k} (K{k_idx+1})",
                    start_l1_k, end_l1_k, dur_k_l1, l1_name_k,
                    q_idx, k_idx, is_l2_hit=use_l2_for_k
                ))
                resource_free_time["MTE2"] = end_l1_k

                dur_k_l0 = self._calc_mte1_cycles(size_k)
                start_l0_k = max(resource_free_time["MTE1"], end_l1_k, l0_slots_k[slot_l0_k])
                end_l0_k = start_l0_k + dur_k_l0
                self.timeline.append(TimelineEvent(
                    "MTE1", f"Load {l0_name_k} (K{k_idx+1})",
                    start_l0_k, end_l0_k, dur_k_l0, l0_name_k, q_idx, k_idx
                ))
                resource_free_time["MTE1"] = end_l0_k
                l1_slots_k[slot_l1_k] = end_l0_k

                if split_type_nb == 'K':
                    sub_k = math.ceil(self.d_base / sub_count_nb)
                    last_mac_end_c1 = end_l0_k
                    for i in range(sub_count_nb):
                        actual_k = min(sub_k, self.d_base - i * sub_k)
                        dur_mac_sub = self._calc_mac_cycles_c1(self.s1_base, self.s2_base, actual_k)
                        if i == 0:
                            start_mac_sub = max(resource_free_time["MAC"],
                                                q_l0_ready_times[q_idx], end_l0_k)
                        else:
                            start_mac_sub = resource_free_time["MAC"]
                        end_mac_sub = start_mac_sub + dur_mac_sub
                        self.timeline.append(TimelineEvent(
                            "MAC", "P", start_mac_sub, end_mac_sub, dur_mac_sub,
                            "L0C", q_idx, k_idx
                        ))
                        resource_free_time["MAC"] = end_mac_sub
                        last_mac_end_c1 = end_mac_sub

                    size_fp = self._calc_fixpipe_p_size()
                    dur_fix_p = self._calc_fixpipe_cycles(size_fp)
                    start_fix_p = max(resource_free_time["FIXPIPE"], last_mac_end_c1)
                    end_fix_p = start_fix_p + dur_fix_p
                    self.timeline.append(TimelineEvent(
                        "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                        "UB", q_idx, k_idx
                    ))
                    resource_free_time["FIXPIPE"] = end_fix_p
                    l0_slots_k[slot_l0_k] = last_mac_end_c1
                    fixpipe_p_ready[k_idx] = end_fix_p

                else:
                    # matmulFull
                    dur_mac_c1 = self._calc_mac_cycles_c1(self.s1_base, self.s2_base, self.d_base)
                    start_mac_c1 = max(resource_free_time["MAC"], q_l0_ready_times[q_idx], end_l0_k)
                    end_mac_c1 = start_mac_c1 + dur_mac_c1
                    self.timeline.append(TimelineEvent(
                        "MAC", "P", start_mac_c1, end_mac_c1, dur_mac_c1,
                        "L0C", q_idx, k_idx
                    ))
                    resource_free_time["MAC"] = end_mac_c1
                    l0_slots_k[slot_l0_k] = end_mac_c1

                    size_fp = self._calc_fixpipe_p_size()
                    dur_fix_p = self._calc_fixpipe_cycles(size_fp)
                    start_fix_p = max(resource_free_time["FIXPIPE"], end_mac_c1)
                    end_fix_p = start_fix_p + dur_fix_p
                    self.timeline.append(TimelineEvent(
                        "FIXPIPE", "P", start_fix_p, end_fix_p, dur_fix_p,
                        "UB", q_idx, k_idx
                    ))
                    resource_free_time["FIXPIPE"] = end_fix_p
                    fixpipe_p_ready[k_idx] = end_fix_p

        # ── Phase 2: V1 for all k in group ──────────────────────────────
        # Note: V1[k0] can overlap with MAC C1[k1] (different hardware)
        for k_idx in k_group:
            dur_v1 = self._calc_vector_v1_cycles()
            start_v1 = max(resource_free_time["VECTOR_V1"], fixpipe_p_ready[k_idx])
            end_v1 = start_v1 + dur_v1
            self.timeline.append(TimelineEvent(
                "VECTOR_V1", "P", start_v1, end_v1, dur_v1,
                "CUBE", q_idx, k_idx
            ))
            resource_free_time["VECTOR_V1"] = end_v1
            v1_ready[k_idx] = end_v1

        # ── Phase 3: C2 for all k in group ──────────────────────────────
        for k_idx in k_group:
            p_ready = v1_ready[k_idx]
            size_p_mte3 = self._calc_mte3_p_size()
            dur_mte3 = self._calc_mte3_cycles(size_p_mte3)
            start_mte3 = max(resource_free_time["MTE3"], p_ready)
            end_mte3 = start_mte3 + dur_mte3
            self.timeline.append(TimelineEvent(
                "MTE3", "P", start_mte3, end_mte3, dur_mte3,
                "CUBE", q_idx, k_idx
            ))
            resource_free_time["MTE3"] = end_mte3

            split_type_nb_c2, sub_count_nb_c2 = self._get_c2_split()
            slot_l1_v = k_idx % len(l1_slots_v)
            slot_l0_v = k_idx % len(l0_slots_v)

            if split_type_nb_c2 == 'N':
                # matmulN for C2: per-sub-tile V load (MTE2 V_sub + MTE1 V_sub) inside loop
                sub_n_c2 = self.baseN_C2
                last_mac_end_c2 = 0.0
                end_l0_v = 0.0
                end_fix_o = 0.0
                for i in range(sub_count_nb_c2):
                    actual_n = min(sub_n_c2, self.d_base - i * sub_n_c2)
                    size_v_sub = self.s2_base * actual_n * self._get_kv_element_size()

                    # MTE2 V_sub[i]: load sub-tile to L1
                    dur_v_sub_l1 = self._calc_mte2_cycles(size_v_sub, use_l2=False)
                    start_l1_v_sub = max(resource_free_time["MTE2"], l1_slots_v[slot_l1_v], p_ready)
                    end_l1_v_sub = start_l1_v_sub + dur_v_sub_l1
                    self.timeline.append(TimelineEvent(
                        "MTE2", f"Load {l1_name_v} (V{k_idx+1}_sub{i})",
                        start_l1_v_sub, end_l1_v_sub, dur_v_sub_l1, l1_name_v, q_idx, k_idx
                    ))
                    resource_free_time["MTE2"] = end_l1_v_sub

                    # MTE1 V_sub[i]: move sub-tile from L1 to L0
                    dur_v_sub_l0 = self._calc_mte1_cycles(size_v_sub)
                    start_l0_v_sub = max(resource_free_time["MTE1"], end_l1_v_sub, l0_slots_v[slot_l0_v])
                    end_l0_v_sub = start_l0_v_sub + dur_v_sub_l0
                    self.timeline.append(TimelineEvent(
                        "MTE1", f"Load {l0_name_v} (V{k_idx+1}_sub{i})",
                        start_l0_v_sub, end_l0_v_sub, dur_v_sub_l0, l0_name_v, q_idx, k_idx
                    ))
                    resource_free_time["MTE1"] = end_l0_v_sub
                    l1_slots_v[slot_l1_v] = end_l0_v_sub
                    end_l0_v = end_l0_v_sub

                    dur_mac_sub = self._calc_mac_cycles_c2(self.s1_base, actual_n, self.s2_base)
                    if i == 0:
                        start_mac_sub = max(resource_free_time["MAC"], end_mte3, end_l0_v_sub)
                    else:
                        # Next sub-MAC can start after previous FIXPIPE completes
                        start_mac_sub = max(resource_free_time["MAC"], end_fix_o, end_l0_v_sub)
                    end_mac_sub = start_mac_sub + dur_mac_sub
                    self.timeline.append(TimelineEvent(
                        "MAC", "O", start_mac_sub, end_mac_sub, dur_mac_sub,
                        "L0C", q_idx, k_idx
                    ))
                    resource_free_time["MAC"] = end_mac_sub
                    last_mac_end_c2 = end_mac_sub

                    # Each sub-tile gets its own FIXPIPE (no L0C accumulation)
                    size_fo_sub = self.s1_base * actual_n * 4  # FP32 output
                    dur_fix_o = self._calc_fixpipe_cycles(size_fo_sub)
                    start_fix_o = max(resource_free_time["FIXPIPE"], end_mac_sub)
                    end_fix_o = start_fix_o + dur_fix_o
                    self.timeline.append(TimelineEvent(
                        "FIXPIPE", "O", start_fix_o, end_fix_o, dur_fix_o,
                        "UB", q_idx, k_idx
                    ))
                    resource_free_time["FIXPIPE"] = end_fix_o

                l0_slots_v[slot_l0_v] = last_mac_end_c2
                fixpipe_o_ready[k_idx] = end_fix_o

            else:
                # For 'K' and 'full': monolithic V load before MAC
                size_v = self._calc_v_size()
                dur_v_l1 = self._calc_mte2_cycles(size_v, use_l2=False)
                start_l1_v = max(resource_free_time["MTE2"], l1_slots_v[slot_l1_v], p_ready)
                end_l1_v = start_l1_v + dur_v_l1
                self.timeline.append(TimelineEvent(
                    "MTE2", f"Load {l1_name_v} (V{k_idx+1})",
                    start_l1_v, end_l1_v, dur_v_l1, l1_name_v, q_idx, k_idx
                ))
                resource_free_time["MTE2"] = end_l1_v

                dur_v_l0 = self._calc_mte1_cycles(size_v)
                start_l0_v = max(resource_free_time["MTE1"], end_l1_v, l0_slots_v[slot_l0_v])
                end_l0_v = start_l0_v + dur_v_l0
                self.timeline.append(TimelineEvent(
                    "MTE1", f"Load {l0_name_v} (V{k_idx+1})",
                    start_l0_v, end_l0_v, dur_v_l0, l0_name_v, q_idx, k_idx
                ))
                resource_free_time["MTE1"] = end_l0_v
                l1_slots_v[slot_l1_v] = end_l0_v

                if split_type_nb_c2 == 'K':
                    sub_k = math.ceil(self.s2_base / sub_count_nb_c2)
                    last_mac_end_c2 = 0.0
                    for i in range(sub_count_nb_c2):
                        actual_k = min(sub_k, self.s2_base - i * sub_k)
                        dur_mac_sub = self._calc_mac_cycles_c2(self.s1_base, self.d_base, actual_k)
                        if i == 0:
                            start_mac_sub = max(resource_free_time["MAC"], end_mte3, end_l0_v)
                        else:
                            start_mac_sub = resource_free_time["MAC"]
                        end_mac_sub = start_mac_sub + dur_mac_sub
                        self.timeline.append(TimelineEvent(
                            "MAC", "O", start_mac_sub, end_mac_sub, dur_mac_sub,
                            "L0C", q_idx, k_idx
                        ))
                        resource_free_time["MAC"] = end_mac_sub
                        last_mac_end_c2 = end_mac_sub

                    size_fo = self._calc_fixpipe_o_size()
                    dur_fix_o = self._calc_fixpipe_cycles(size_fo)
                    start_fix_o = max(resource_free_time["FIXPIPE"], last_mac_end_c2)
                    end_fix_o = start_fix_o + dur_fix_o
                    self.timeline.append(TimelineEvent(
                        "FIXPIPE", "O", start_fix_o, end_fix_o, dur_fix_o,
                        "UB", q_idx, k_idx
                    ))
                    resource_free_time["FIXPIPE"] = end_fix_o
                    l0_slots_v[slot_l0_v] = last_mac_end_c2
                    fixpipe_o_ready[k_idx] = end_fix_o

                else:
                    # matmulFull
                    dur_mac_c2 = self._calc_mac_cycles_c2(self.s1_base, self.d_base, self.s2_base)
                    start_mac_c2 = max(resource_free_time["MAC"], end_mte3, end_l0_v)
                    end_mac_c2 = start_mac_c2 + dur_mac_c2
                    self.timeline.append(TimelineEvent(
                        "MAC", "O", start_mac_c2, end_mac_c2, dur_mac_c2,
                        "L0C", q_idx, k_idx
                    ))
                    resource_free_time["MAC"] = end_mac_c2
                    l0_slots_v[slot_l0_v] = end_mac_c2

                    size_fo = self._calc_fixpipe_o_size()
                    dur_fix_o = self._calc_fixpipe_cycles(size_fo)
                    start_fix_o = max(resource_free_time["FIXPIPE"], end_mac_c2)
                    end_fix_o = start_fix_o + dur_fix_o
                    self.timeline.append(TimelineEvent(
                        "FIXPIPE", "O", start_fix_o, end_fix_o, dur_fix_o,
                        "UB", q_idx, k_idx
                    ))
                    resource_free_time["FIXPIPE"] = end_fix_o
                    fixpipe_o_ready[k_idx] = end_fix_o

        # ── Phase 4: V2 for all k in group ──────────────────────────────
        for k_idx in k_group:
            dur_v2 = self._calc_vector_v2_cycles()
            start_v2 = max(resource_free_time["VECTOR_V2"], fixpipe_o_ready[k_idx])
            end_v2 = start_v2 + dur_v2
            self.timeline.append(TimelineEvent(
                "VECTOR_V2", "O", start_v2, end_v2, dur_v2,
                "CUBE", q_idx, k_idx
            ))
            resource_free_time["VECTOR_V2"] = end_v2

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

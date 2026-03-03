
# PRELOAD Workspace重构计划

## 需求分析

### 1. L0C缓存区管理
- **当前问题**：MAC开始时没有检查L0C是否有足够空间，可能导致数据覆盖
- **需求**：MAC需要等待L0C有足够空间，否则等待FIXPIPE把上一块搬出
- **L0C容量**：256KB，默认开启doublebuffer（2个128KB槽位）
- **基本块大小**：128×128×4 = 64KB（FP32）

### 2. PRELOAD Workspace管理
- **UB容量**：256KB
- **preload1模式**：2个workspace，每个64KB
- **preload2模式**：3个workspace，每个64KB
- **不同WS间无依赖**：即使指令发射有先后，实际流水线应紧密排布
- **预期效果**：连续3个K1K2K3搬运，无间隔

### 3. L1内存约束
- **L1容量**：512KB
- **基本块大小**：
  - Q: 128×128×1 = 16KB (FP8)
  - K: 256×128×1 = 32KB (FP8)
  - V: 256×128×1 = 32KB (FP8)
  - P: 128×256×4 = 128KB (FP32)

---

## 重构计划表

| 阶段 | 任务 | 优先级 | 状态 | 说明 |
|------|------|--------|------|------|
| 1 | 分析当前L0C管理逻辑 | 高 | 待开始 | 检查MAC启动条件 |
| 2 | 实现L0C缓存区追踪 | 高 | 待开始 | 记录每个槽位的占用情况 |
| 3 | 修改MAC启动条件 | 高 | 待开始 | 等待L0C有足够空间 |
| 4 | 实现Workspace管理器 | 高 | 待开始 | 管理WS分配和释放 |
| 5 | 修改PRELOAD流水线逻辑 | 高 | 待开始 | 支持WS间无依赖 |
| 6 | 更新reference.md | 中 | 待开始 | 记录L0C和WS约束 |
| 7 | 编写测试用例 | 中 | 待开始 | 验证L0C和WS逻辑 |
| 8 | 性能测试和优化 | 低 | 待开始 | 对比重构前后性能 |

---

## 详细设计

### 阶段1：分析当前L0C管理逻辑

**目标**：理解当前MAC启动条件

**检查点**：
- MAC启动时是否检查L0C槽位状态？
- L0C槽位是否记录占用/释放时间？
- FIXPIPE完成后是否释放L0C槽位？

### 阶段2：实现L0C缓存区追踪

**目标**：记录每个L0C槽位的占用情况

**设计**：
```python
class L0CSlotTracker:
    def __init__(self, num_slots: int, slot_size: int):
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.slots = [{'free_time': 0.0, 'owner': None} for _ in range(num_slots)]
    
    def allocate(self, block_id: str, size: int, current_time: float) -> Tuple[int, float]:
        """分配L0C槽位，返回槽位索引和就绪时间"""
        pass
    
    def release(self, slot_idx: int, release_time: float):
        """释放L0C槽位"""
        pass
```

### 阶段3：修改MAC启动条件

**目标**：MAC等待L0C有足够空间

**修改点**：
- MAC启动条件增加：`max(..., l0c_ready_time)`
- l0c_ready_time = L0CSlotTracker.allocate()

### 阶段4：实现Workspace管理器

**目标**：管理WS分配和释放

**设计**：
```python
class WorkspaceManager:
    def __init__(self, total_size: int, ws_count: int):
        self.total_size = total_size  # 256KB
        self.ws_count = ws_count        # 2 or 3
        self.ws_size = total_size // ws_count  # 64KB or 85KB
        self.workspaces = [{'free_time': 0.0, 'owner': None} for _ in range(ws_count)]
    
    def allocate(self, block_id: str, current_time: float) -> Tuple[int, float]:
        """分配workspace，返回WS索引和就绪时间"""
        pass
    
    def release(self, ws_idx: int, release_time: float):
        """释放workspace"""
        pass
```

### 阶段5：修改PRELOAD流水线逻辑

**目标**：支持WS间无依赖，区分default， preload=1，和 preload=2， 删除N-buffer

**修改点**：
- PRELOAD模式：使用WorkspaceManager管理WS
- C1阶段：立即发射，不等前一个WS释放
- V1阶段：分配WS，立即执行
- C2阶段：立即发射，不等前一个WS释放
- V2阶段：释放WS，立即执行

### 阶段6：更新reference.md

**目标**：记录L0C和WS约束

**更新内容**：
- 新增L0C缓存区管理约束
- 新增PRELOAD workspace管理约束
- 更新流水线执行顺序说明

### 阶段7：编写测试用例

**目标**：验证L0C和WS逻辑

**测试用例**：
- `test_l0c_allocation`: 测试L0C槽位分配
- `test_l0c_release`: 测试L0C槽位释放
- `test_workspace_allocation`: 测试WS分配
- `test_workspace_release`: 测试WS释放
- `test_preload_pipeline`: 测试PRELOAD流水线
- `test_mac_wait_for_l0c`: 测试MAC等待L0C

### 阶段8：性能测试和优化

**目标**：对比重构前后性能

**测试用例**：
- 测试不同参数组合下的性能
- 对比重构前后的总周期数
- 分析瓶颈单元

---

## 实施步骤

### 第一步：分析当前代码
- 检查MAC启动条件
- 检查L0C槽位管理
- 检查PRELOAD流水线逻辑

### 第二步：实现L0C追踪器
- 创建L0CSlotTracker类
- 实现allocate和release方法
- 集成到现有代码

### 第三步：实现Workspace管理器
- 创建WorkspaceManager类
- 实现allocate和release方法
- 集成到PRELOAD模式

### 第四步：修改流水线逻辑
- 修改MAC启动条件
- 修改PRELOAD流水线逻辑
- 确保无依赖关系

### 第五步：编写测试用例
- 创建test_l0c.py
- 创建test_workspace.py
- 创建test_preload_pipeline.py

### 第六步：性能测试
- 运行测试用例
- 对比性能数据
- 优化瓶颈

---

## 预期效果

### L0C管理改进
- MAC启动时正确等待L0C有足够空间
- 避免数据覆盖
- 提高流水线效率

### PRELOAD Workspace管理改进
- WS间无依赖，紧密排布
- 减少流水线间隔
- 提高整体吞吐量

### 性能提升预期
- 总周期数减少10-20%
- MAC利用率提升
- 流水线更加紧密

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| L0C追踪引入性能开销 | 低 | 低 | 优化数据结构 |
| Workspace管理复杂度增加 | 中 | 中 | 详细测试验证 |
| 流水线逻辑破坏 | 中 | 高 | 保留原有逻辑作为fallback |
| 测试覆盖不足 | 中 | 中 | 增加测试用例 |

---

## 进度记录

| 时间 | 阶段 | 进度 | 备注 |
|------|------|------|------|
| 2026-03-02 | 计划制定 | 100% | 完成重构计划表 |

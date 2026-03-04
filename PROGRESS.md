# 重构进度：L0C 槽位追踪 + PRELOAD UB Workspace 管理

> 完整计划详见 `/home/developer/.claude/plans/vivid-squishing-otter.md`

## 背景
- **L0C 槽位约束**：MAC 启动前须等待 L0C 有空闲槽位（doublebuffer 2槽，每槽存1个完整基本块）
- **UB Workspace 管理**：PRELOAD_1=2WS，PRELOAD_2=3WS（各64KB），WS 在 FIXPIPE-P 写入时占用，V2 完成后释放
- **枚举更新**：`PRELOAD` → `PRELOAD_1`，新增 `PRELOAD_2`
- **N_BUFFER 已删除**：移除核间流水 N_BUFFER 模式及相关代码

---

## 进度跟踪

### 枚举 & 新类
- [x] Step 1：`core/enums.py` — PRELOAD → PRELOAD_1，新增 PRELOAD_2，删除 N_BUFFER
- [x] Step 2：新增 `L0CSlotTracker` 类（c1_modeler.py 顶部）
- [x] Step 3：新增 `UBWorkspaceTracker` 类（c1_modeler.py）
- [x] Step 10：全局替换所有 PRELOAD → PRELOAD_1 引用（同时删除 N_BUFFER 所有引用）

### L0C 集成
- [x] Step 4：`run_simulation()` 初始化 `l0c_tracker = L0CSlotTracker(2)`
- [x] Step 5a：`_process_k_blocks`（DEFAULT 路径）— matmulFull/N/K C1+C2 MAC 前 allocate，FIXPIPE 后 release
- [x] Step 5b：`_process_c1_only`（PRELOAD 路径 C1）— 三路 MAC 前 allocate，FIXPIPE 后 release；新增 `ws_avail_time` 参数
- [x] Step 5c/7：`_process_c2_stage` — 三路 MAC-O 前 allocate，FIXPIPE-O 后 release；返回 `end_vector_v2`
- ~~Step 5d~~：N_BUFFER 已删除，无需集成

### PRELOAD 重构
- [x] Step 6：`_process_c1_only` 新增 `ws_avail_time` 参数（MTE2 开始时间下界）
- [x] Step 7：`_process_c2_stage` 返回 `end_v2`
- [x] Step 8：PRELOAD_1 调度循环重构（含 `UBWorkspaceTracker`，2WS，C2_DELAY=1）
- [x] Step 9：PRELOAD_2 新调度分支（3WS，C2_DELAY=2，V1 紧跟 C1）

### 测试
- [x] Step 11a：`tests/test_l0c_tracker.py` — L0CSlotTracker 单元测试
- [x] Step 11b：`tests/test_ub_workspace_tracker.py` — UBWorkspaceTracker 单元测试
- [x] Step 11c：`tests/test_preload2_pipeline.py` — PRELOAD_2 专项测试
- [x] Step 11d：`test_preload_pipeline.py`、`test_c1_modeler_basic.py` 已是最新（无需改动）
- [x] Step 11e：`tests/test_l0c_integration.py` — L0C 集成测试
- [x] 全量测试通过：`python -m pytest tests/ -q`（**108 passed**）

---

## 验证命令

```bash
python -m pytest tests/ -q                                         # 全量（当前：108 passed）
python -m pytest tests/test_l0c_tracker.py -v                    # L0C 单元
python -m pytest tests/test_ub_workspace_tracker.py -v           # WS 单元
python -m pytest tests/test_preload2_pipeline.py -v              # PRELOAD_2
python -m pytest tests/test_l0c_integration.py -v                # L0C 集成
python examples/test_preload.py                                   # 性能对比
```

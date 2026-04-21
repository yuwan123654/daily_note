# Daily Note — 2026-04-21

## 今日工作概览

今天的工作围绕两条主线：**RAG knowledge base 的最终补充**，以及**训练数据集 `joint_master_records_with_verdict.jsonl` 的 gap 分析与补全方案设计**，同时深入讨论了 rollback decision 的训练逻辑与 r_reflect 信号的关系。

---

## 一、RAG Knowledge Base 最终补充

### 1.1 本次新增 chunk 来源

在已有 305 条 chunk 的基础上，今天分三批追加：

**第一批（Nsight 工具文档，9 chunks）**

从 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 和 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 抓取页面内容，切成：
- Nsight Compute 概述（Section Sets / basic / full）
- Memory Workload Analysis（Mem Busy / Max Bandwidth / Mem Pipes Busy 三种瓶颈）
- Occupancy & Launch Metrics（launch__ 系列指标）
- Warp Stalls & Scheduler（stall_barrier / stall_long_scoreboard / stall_mio_throttle）
- Roofline Charts（compute-bound vs memory-bound 判定）
- Replay Modes & Overhead（Kernel / Application / Range Replay）
- Nsight Systems 概述（timeline view / CUDA API tracing）
- Timeline / Transfer / Overlap 分析（kernel 执行 / H2D/D2H / CPU-GPU overlap / gap 分析）
- Nsight Systems CLI（nsys profile 关键选项）

**第二批（PyTorch Profiler & Benchmark，7 chunks）**

从 [torch.profiler API docs](https://docs.pytorch.org/docs/stable/profiler.html) 和 TensorBoard tutorial 整合，切成：
- torch.profiler 基本用法（activities / record_shapes / profile_memory / with_stack）
- Schedule 多轮采集（wait/warmup/active/repeat / tensorboard_trace_handler）
- 输出分析（key_averages().table() / sort_by / export_chrome_trace / export_memory_timeline）
- record_function 自定义标注（与 NVTX 的类比 / CUDA kernel 归因）
- Memory Profiling（cpu/cuda_memory_usage / snapshot / 常见内存问题）
- torch.utils.benchmark Timer（blocked_autorange / CUDA 同步 / Measurement 统计）
- Benchmark Best Practices（min_run_time / GPU 时钟锁定 / Fuzzer / Compare）

**第三批（CUDA Runtime API 重点章节，9 chunks）**

从上传的 CUDA_Runtime_API.pdf（v13.2，717 页）提取章节体文本，切成：
- API synchronization behavior（memcpy 同步/异步 5 条规则）
- Stream synchronization behavior（legacy vs per-thread default stream）
- Graph object thread safety（cudaGraph_t 不内部同步，必须外部串行化）
- Rules for version mixing（ABI 按 major release 绑定，不能跨版本传递类型）
- Stream Management API（Create/Destroy/Synchronize/Query/WaitEvent/BeginCapture）
- Event Management API（Create/Record/Synchronize/ElapsedTime / DisableTiming flags）
- Execution Control API（cudaLaunchKernel / CooperativeKernel / FuncSetAttribute）
- Occupancy Calculation API（MaxActiveBlocksPerMultiprocessor / MaxPotentialBlockSize）
- Stream Ordered Memory Allocator（cudaMallocAsync/cudaFreeAsync / memory pool）

### 1.2 Gap 分析结论与补充 chunks

对上传的 `filtered_chunks.jsonl`（305 条）运行覆盖评估，发现：

**原始弱项（1-2 chunks）**：`cpp_extension`（2）、`roofline`（2）  
**原始缺失（0 chunks）**：`compile_error_patterns`、`shape_mismatch`

针对以上缺口，手写 9 个补充 chunk：
- CUDA 常见编译错误（8 类，含 cause + fix）
- PyTorch Extension 构建错误（8 类，ABI / arch / TORCH_LIBRARY 重复等）
- Contiguous tensor 陷阱（transpose/permute/expand/slice 导致非连续）
- dtype mismatch（data_ptr<float>() 假设 / AT_DISPATCH 宏族）
- shape mismatch（6 类模式 / TORCH_CHECK 标准检查清单）
- TORCH_CHECK / assertions（C10_CUDA_KERNEL_LAUNCH_CHECK / compute-sanitizer）
- 数值精度问题（fast math / fp16 溢出 / NaN 传播 / reduction order）
- torch.utils.cpp_extension 完整用法（setup.py / load_inline / TORCH_CUDA_ARCH_LIST）
- Roofline 解读指南（4 类 kernel 位置的诊断与优化策略）

**追加 PTX / cuDNN / CUB 专项 chunk（12 chunks）**：
- PTX ISA 概述（寄存器类型 / state spaces / special registers / 关键指令类）
- Inline PTX（asm() 语法 / constraint letters / volatile / 5 个使用场景）
- CUDA Binary Utils（cuobjdump -sass/-res-usage / nvdisasm / SASS 分析）
- cuDNN ConvFwd/BwdData/BwdFilter（NHWC 要求 / Support Surface 限制）
- cuDNN Normalization（BatchNorm/LayerNorm/RMSNorm/InstanceNorm 引擎支持）
- cuDNN Fused Attention（Flash Attention / Stream-K / head dim 约束）
- cuDNN Execution Plan（OperationGraph→EngineHeur→EngineCfg→ExecutionPlan→VariantPack）
- cuDNN Layout & Alignment 约束（NHWC 强制 / 三档 alignment / virtual tensor / 5 个最常见 not supported 原因）
- CUB 概述（Thread/Warp/Block/Device 四层架构）
- CUB BlockReduce & BlockScan（WARP_REDUCTIONS/RAKING/WARP_SCANS 算法变体）
- CUB Warp Primitives（WarpReduce/WarpScan/WarpMergeSort / shuffle-based 无同步）
- CUB Device Primitives（两阶段 query-execute / DeviceReduce/Scan/RadixSort）

### 1.3 最终 RAG 数据库状态

| 指标 | 数值 |
|---|---|
| filtered_chunks.jsonl 总条数 | **326 条** |
| 总词数 | **174,339 词** |
| 来源数量 | 9+ 个文档来源 |
| 核心主题覆盖 | 32/32（全部 ≥ 1 chunk，30/32 ≥ 3 chunk） |

---

## 二、数据集 Gap 分析（joint_master_records_with_verdict.jsonl）

### 2.1 数据基本情况

- 总记录数：3621 条
- sample_type 分布：performance_optimization 2174 / correctness_failure 797 / stagnation_or_accept 289 / compile_failure 361
- verdict teacher 已补全顶层字段（diagnosis / decision / confidence / observations / patch_spec.patch_goal）
- 所有 record 均有 optimized_code

### 2.2 核心 Gap 汇总

**P0 — 阻塞性缺口：**

**① Decision taxonomy 完全错配**
- 当前数据：`optimize / reject_and_patch / accept`（3 值）
- Workflow Phase 1 & 5 要求：`local_patch / rewrite_kernel / revise_host / rollback / switch_strategy / replan / stop`（7 值）
- 影响：Phase 1 SFT 和 Phase 5 DPO 均无法直接使用，3621 条全部需要重标

**② Verifier policy-level CoT 完全缺失**
- `## Reasoning` 段在所有记录里都为空
- 这是论文 C2 贡献的核心 novelty，缺失意味着 Phase 1 SFT 训练格式根本不符合设计

**③ Coder action-level CoT 完全缺失**
- `coder_target.optimized_code` 是纯代码，无 `## Thought` 思考过程
- Phase 2B feedback-to-patch 训练格式要求先有 CoT 再给代码

**P1 — 质量性缺口：**

**④ perf_type 分布严重偏斜**

| perf_type | 实际 | 推荐 | 偏差 |
|---|---|---|---|
| bad_parallelization | 65.9% | 15% | 4.4× 过多 |
| bad_library_path | 0.1% | 20% | 200× 不足 |
| descriptor_or_workspace_rebuild | 0.1% | 20% | 200× 不足 |
| redundant_memory_ops | 0.5% | 15% | 30× 不足 |
| unfused_post_ops | 1.7% | 15% | 9× 不足 |

**⑤ failure_type 分布偏斜**
- semantic_mismatch 占 correctness 类的 96%（762/797）
- shape_contract_mismatch / dtype_contract_mismatch / missing_init_args 极少

**⑥ rag_retrieval 缺失 902 条（25%）**：correctness 类缺 219 条，perf 类缺 362 条

**⑦ feedback_text 缺失 613 条**（is_trainable=True 的记录里）

**P2 — 结构性缺口：**

**⑧ Stagnation 样本对 Coder 完全不可用**
- 289 条全部 `is_trainable_for_coder = False`，`pair_type = None`
- Coder 在 `stop` decision 下应该学会"不产生修改"，但负例数据完全缺失

**P3 — 流程依赖性缺口：**

**⑨ Phase 5 DPO 所需轨迹字段全部缺失**
- `trajectory_history / step_idx / best_speedup_so_far / candidate_decision alternatives` 均为空
- 这些来自 Phase 3 闭环采样，是流程硬依赖，无法从静态数据补

### 2.3 补充优先级

| 优先级 | 缺口 | 估计工作量 |
|---|---|---|
| P0 | decision 7 值重标 | 中（teacher diff 推断） |
| P0 | Verifier CoT 合成 | 中 |
| P0 | Coder CoT 合成 | 中 |
| P1 | perf_type 偏斜修正 | 大 |
| P1 | feedback_text 补全 | 小 |
| P1 | rag_retrieval 补全 | 小 |
| P2 | failure_type 分布调整 | 中 |
| P2 | stagnation Coder 负例构造 | 中 |
| P3 | Phase 3 轨迹采样 | 大（工程） |

---

## 三、Rollback Decision 训练设计

### 3.1 Rollback 的充要触发条件

rollback ≠ "当前代码比 PyTorch 慢"，而是需要同时满足：

- **条件 A**：当前代码（B）比轨迹历史中某个具体版本（A）更差（存在可回溯的历史最好点）
- **条件 B**：从 B 向前修复的代价高于抛弃 B 回到 A（继续在 B 上叠加工作是错误方向）

**典型场景**：

| 场景 | 说明 |
|---|---|
| 功能性退步 | turn t+1 的修改破坏了 correctness，改动面广难以反向 patch |
| 性能灾难性退步 | speedup 1.4 → 0.02，B 结构已偏离 A 太远，local_patch 无效 |
| 错误方向累积 | 连续几轮 local_patch 让代码越来越复杂，rollback 成本低于 rewrite |

**不应触发 rollback 的情况**：历史上从来没有更好过（无 A 可回溯）→ rewrite_kernel；B 和 A 差不多只是参数错了 → local_patch；整体路线判断错误，A 也不值得回去 → switch_strategy / replan

### 3.2 训练数据构造：三条路线

**路线 A：Phase 3 轨迹挖掘（语义最正确，需等待）**
- 从真实多轮轨迹中挖掘 turn t 比 turn t-1 退步的情况
- 完全真实，prior_code 是 Coder 真实产生的；但需等 Phase 3 跑完

**路线 B：合成轨迹注入（现在能做）**
- 原材料：1676 条（candidate speedup < 0.5，optimized speedup > 1.2，optimized correct=True）
- 把 `optimized_code` 放入 `trajectory_history` 的 step 0 充当 prior_code A
- 把 `candidate_code` 当作失败尝试 B
- 构造 rollback verdict + Coder target（= optimized_code 原文）
- 建议生产量：100-150 条（不超过总数据量 5%）

**路线 C：受控突变（质量最高，需执行环境）**
- 对 optimized_code 施加可控坏突变（改 block size 为不对齐值 / 把 shared 改 global 等）
- 执行验证确认退步后得到干净 A/B 对

### 3.3 关键实现约束

- **Coder rollback target 必须与 trajectory_history 里的 prior_code 高度重叠**（几乎一致），才能让模型区分 rollback 和 rewrite_kernel
- **Verifier patch_spec 只写回退指令**（`"return step N implementation verbatim"`），不附加优化建议
- 语义区别：`rewrite_kernel` = 丢弃 B 生成新代码；`rollback` = 从 context 里找回历史版本 A

### 3.4 Rollback 与 r_reflect 的关系

**表面冲突**：rollback 执行后 speedup 最多回到 A（= `best_up_to_t`），所以 `r_reflect = max(best_future - best_up_to_t, 0)` 对 rollback 几乎永远是 0，training signal 消失。

**实质**：rollback 和 r_reflect 不根本冲突，但存在**系统性低估**问题，来自 `best_up_to_t` 的基准选取错误。

当前公式用历史全轨迹最大 speedup 作为 rollback 的基准，而这恰好是 A 的 speedup。rollback 之后达到 A，r_reflect = 0。这使 rollback 在层 2 DPO 里拿不到任何直接偏好信号。

**修复方案**：对 rollback decision 特判，使用**当前被回滚的坏版本 B 的 speedup 作为基准**：

```python
def compute_r_reflect(trajectory, step_idx, window=2):
    current_decision = steps[step_idx].verifier_output.decision
    
    if current_decision == "rollback":
        # rollback 的价值是从坏状态（B）逃出来，基准用 B 的 speedup
        baseline = steps[step_idx].executor_result.speedup or 0
    else:
        baseline = max([s.speedup or 0 for s in steps[:step_idx+1]])
    
    future = [s.speedup for s in steps[step_idx+1:step_idx+1+window] if s.speedup]
    best_future = max(future) if future else baseline
    return max(best_future - baseline, 0)
```

这样 rollback 之后只要后续超过 B（通常很容易，B 很差），就得到正向 r_reflect 信号，合理反映"rollback 的价值是逃出坏状态"。

**结论**：rollback 与 r_reflect 的设计意图不冲突（r_reflect 奖励的是"rollback 之后轨迹变好"），但实施时 `best_up_to_t` 的计算需要对 rollback 做特判，否则训练信号几乎消失。这个修改很小但影响决定性。

---

## 四、关键文件状态

| 文件 | 状态 | 备注 |
|---|---|---|
| `filtered_chunks.jsonl` | ✓ 326 条，174,339 词 | RAG knowledge base 完成 |
| `joint_master_records_with_verdict.jsonl` | △ 3621 条，verdict 顶层已填 | decision 需重标为 7 值，CoT 待合成 |
| `dataset_supplement_workflow_summary.md` | ✓ 已生成 | 本次工作流总结 |
| `verdict_completion_plan.md` | ✓ 已生成 | Teacher 补全 4-stage pipeline 设计 |

---

## 五、下一步行动

### 立即可做
1. **decision 7 值重标**：写 teacher prompt，从 candidate/optimized diff 推断 7 值 decision，先跑 10 条验证 prompt 质量
2. **Verifier CoT 合成**：设计 policy-level reasoning 模板（trajectory history 分析 → strategy 对比 → decision 推理），用 teacher 批量生成
3. **Coder CoT 合成**：从 optimized_code 反向生成"为何这样改"的思路，作为 `## Thought` 段

### 需要规划
4. **perf_type 偏斜修正**：从 2174 条 perf 样本里把符合 `bad_library_path / descriptor / redundant` diff 特征的重新标注
5. **rollback 训练数据**：从 1676 条原材料里筛选（edit_distance_bucket=large），用路线 B 造 100-150 条合成数据
6. **Phase 3 闭环采样**：规划时间节点，这是 Phase 5 DPO 的硬依赖

### 待设计
7. **r_reflect rollback 特判**：在 `compute_r_reflect` 里加 rollback 分支，避免 training signal 消失
8. **stagnation Coder 负例**：设计 `stop` decision 下 Coder 的目标格式（保守输出 / 不修改）

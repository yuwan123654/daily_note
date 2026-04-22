# Daily Note — 2026-04-22

## 今日工作概览

今天围绕三条主线：**decision 重标与 CoT 合成规划**、**perf_type 偏斜修正可行性分析**、**Skill 2 数据工程（schema 定稿 + Skill 1 格式修复 + 混合比例 + CoT 全量生成）**。

---

## 一、Decision 重标与 CoT 合成规划

### 1.1 规划文档产出

输出文件：`decision_relabel_and_cot_synthesis_plan.md`

核心发现：
- Decision 重标：规则可处理约 90%，Teacher 只处理约 10% 歧义案例
- 最大歧义区（`bad_parallelization` 1432 条 + `memory_bound_access` 458 条）通过四级信号链（patch_goal 关键词 → `__init__` 变化 → kernel 数量变化 → edit 粒度）全部确定性解决，实际零歧义
- CoT 合成：改写而非从零生成，现有 `feedback_text`（avg 308 词）已是素材，只需重组为推理叙事并强制加入 strategy comparison 段

### 1.2 relabel_decision_rules.py v1（初版）

产出：`relabel_decision_rules.py`（Rule 1-4，全规则覆盖）

规则体系：
- Rule 1：`stagnation_or_accept` → `stop`（289 条，100% 确定）
- Rule 2：`compile_failure` → `local_patch`（361 条，100% 确定）
- Rule 3a/3b：`correctness_failure` 按 `failure_type` → `local_patch` / `rewrite_kernel`
- Rule 4.1-4.4：`performance_optimization` 按 `perf_type` + 优先级信号链

首次运行结果：3621 条全覆盖，所有 consistency check PASS。

最终 7 值分布：`rewrite_kernel` 2631 / `local_patch` 420 / `stop` 289 / `revise_host` 256 / `switch_strategy` 25

---

## 二、是否需要 Teacher 标注

对所有字段做完整 audit，结论：

| 字段 | 缺失量 | 正确处理方式 |
|---|---|---|
| `verifier_cot` | 3621 | Teacher 合成（CoT 规划阶段工作） |
| `coder_cot` | 3621 | 规则模板 + Teacher 补 reason（今天完成规则部分） |
| `rag_retrieval` | 902 | 跑 RAG 检索补填，不用 Teacher |
| `feedback_text` | 613 | 模板拼接生成，不用 Teacher |
| `must_change` | 289 | 空列表是正确语义（stop decision），不填 |

**真正需要 Teacher 的只有 verifier_cot 和 coder_cot 的 brief_reason 部分**，其余字段用非模型方法处理。

---

## 三、perf_type 偏斜修正可行性分析

三类目标 perf_type 的可行性：

**`bad_library_path`：可规则重标，69 条**

三信号联合触发（缺一不满足）：
1. `optimized_code` 出现高层库 API（`torch.nn.functional` / `at::native` / `cublasSgemm` 等）
2. `candidate_code` 的 `__global__` kernel 数量减少（kernel 被移除）
3. `patch_goal` 或 `must_change` 含 `library / native / cublas` 等词

**`redundant_memory_ops`：规则几乎无法可靠识别**

根本原因：`.contiguous()` 在 PyTorch 里是正常调用，不代表冗余；`cpu_gpu_roundtrip` 信号在数据里只有 2 条。这类问题在原始数据里本来就少，不是标注错误。

**`descriptor_or_workspace_rebuild`：规则完全失效，需要 teacher**

已知 3 条真实样本的 `forward()` 是 `return self.conv3d(x)` 一行，开销发生在 `nn.Conv3d` 内部（cuDNN 每次 forward 重新做算法选择），无法从代码层检测。所有文本信号（`caching`、`handle`、`descriptor`）在数据里全是误报。

**根本结论**：`descriptor_or_workspace_rebuild` 和 `redundant_memory_ops` 的严重低估是数据分布问题，不是标注问题，应在 Phase 3 闭环采样时从源头解决，而不是重标现有数据。

---

## 四、relabel_decision_rules.py v2（加入 Rule 5）

### 4.1 Rule 5：bad_library_path perf_type 修正

在 v1 基础上新增 Rule 5，将 69 条误标为 `bad_parallelization` / `memory_bound_access` 的样本改为 `bad_library_path`。

**附带修正 3 条 decision 误判**：这 3 条被 Rule 4.2b（`__init__` 变化 + host 关键词）误判为 `revise_host`，但实际是换库路径，应为 `rewrite_kernel`。Rule 5 检测到 `bad_library_path` 后同步修正。

### 4.2 新增字段

被 Rule 5 修改的 69 条额外写入：
- `verdict.perf_type_v1`：原 perf_type 备份
- `verdict.perf_type_rule`：来源标记（`rule_5_bad_library_path`）

Rule 5 后 bad_library_path：2 → **71**，bad_parallelization：1432 → **1376**

运行结果：8 项 consistency check 全部 PASS。

---

## 五、Evidence Card 追加（merged_cards_v2.jsonl）

基于 `proposed_new_evidence_cards.md` 的 7 张新卡，追加到 `merged_cards_v1.jsonl`（312 条），产出 `merged_cards_v2.jsonl`（319 条）。

联网检索了 CUDA warp-level primitives、`__shfl_down_sync` 误用、PyTorch `load_inline` 官方文档、host/device 执行空间限制等技术细节，充实每张卡的 `pattern_description` 和 `snippet`。

| card_id | 诊断类型 | 主题 |
|---|---|---|
| `seed_semantic_pattern_001` | correctness_error | 错误索引映射 / 坐标变换 |
| `seed_semantic_pattern_002` | correctness_error | 任务替换污染（wrong-task substitution） |
| `seed_semantic_pattern_003` | correctness_error | warp/block 归约同步错误 |
| `seed_compile_pattern_001` | compile_error | load_inline wrapper 契约不匹配 |
| `seed_compile_pattern_002` | compile_error | 未声明符号 / API 签名错误 |
| `seed_compile_pattern_003` | compile_error | 非移植向量类型 / cooperative groups |
| `seed_compile_pattern_004` | compile_error | host ATen API 嵌入 device 代码 |

所有新卡字段完整（14 个必填字段全部覆盖），每卡含 10-12 个 `retrieval_keywords`、4 条 `common_symptoms`、3 条 `must_change_hints`。

---

## 六、Skill 1/2 训练策略讨论

### 6.1 enable_thinking 的正确设置

**结论：`enable_thinking=False`**，但在 SFT 数据里显式包含 `## Thought` 段。

两件事需要区分：
- `enable_thinking=True` → 模型产生隐式 `<think>` token，不受 SFT 监督，推理成本高
- Skill 2 的 CoT → 显式 `## Thought` 文本直接出现在输出序列中，由 SFT loss 直接监督

Skill 2 的 `## Thought` 内容是 Verifier patch_spec 的结构化改写，低熵、模板固定，不需要模型自由扩展思考，`enable_thinking=False` + 显式 CoT 监督是最干净的做法。

### 6.2 Skill 2 训练参数

数据总量 5400 条（Skill 2 with CoT 3600 + Skill 1 without CoT 1800），推荐参数：

| 参数 | 值 | 备注 |
|---|---|---|
| lr | 5e-6 | 比 Skill 1 低一档 |
| epoch | 2 | 不超过 3 |
| effective batch | 8~16 | 梯度累积实现 |
| max_seq_len | 8192~12288 | 代码序列长 |
| warmup_ratio | 0.05 | continual training |
| loss_mask | output_only | Thought 段也在 loss 内 |
| weight_decay | 0.01~0.1 | 防 Skill 1 能力退化 |

训练后验证两个核心检查点：
1. Skill 1 回归：20 条无 current_code 输入 → 确认不输出 `## Thought`
2. Skill 2 格式稳定性：20 条 Skill 2 输入 → `## Thought` 必须引用 diagnosis/decision 实质内容

---

## 七、Skill 2 数据 Schema 定稿

输出文件：`skill2_data_schema.md`

核心设计：

**Prompt 构造材料**（全部来自 `joint_master_records.with_7decision.jsonl`）：
- `candidate_code`：待修改代码
- `reference_task_code`：正确答案参照
- `execution_evidence`：去掉 `raw_artifacts` 后的执行证据
- `verdict`：使用 7 值 decision，含完整 `patch_spec`

**训练目标**：
- `coder_target.coder_cot`：合成字段（今天完成）
- `coder_target.revised_code`：来自 `optimized_code`（stop 时为 `candidate_code`）

**格式控制**：
- `enable_thinking: false`
- `loss_mask: output_only`

**样本筛选**：`stop / rollback / replan` 排除出主训练集（stop 另行处理，后两者需要轨迹数据）。实际可用 3332 条。

---

## 八、Skill 1 数据格式修复与混合

### 8.1 格式问题诊断（qwen3_cuda_sft_cleaned.jsonl，8633 条）

发现三处不对齐：

| 问题 | 实际 | 需要改为 |
|---|---|---|
| system prompt | `You are an expert in PyTorch and CUDA programming.` | `You are a CUDA kernel optimization expert.` |
| assistant 开头 | 直接 ` ```python ` | `## Code\n```python` |
| user 结构 | 散文指令（无 `##` 标记） | 可接受，输入特征足够区分 |

### 8.2 Token 不平衡问题（更关键）

表面 2:1 记录比，实际 token 比：

| Skill 1 条数 | Token 比 S2:S1 | 建议 |
|---|---|---|
| 1800（原计划） | 8.5:1 | ❌ 遗忘风险高 |
| 4500 | 3.2:1 | ✓ 推荐 |
| 8633（全量） | 1.3:1 | ✓✓ 最稳 |

原因：Skill 2 每条含两份代码（current_code + revised_code）约 5000 tokens，Skill 1 只有约 1400 tokens，记录数比无法反映实际梯度贡献比。

### 8.3 mix_skill1_skill2.py

产出：`mix_skill1_skill2.py`

功能：
- 自动修复 Skill 1 三处格式问题
- 按指定条数采样 Skill 1（`--skill1-n`）
- 混合并 shuffle，输出带 `skill` 字段的 JSONL
- `--save-fixed-skill1` 保存格式修复后的完整 Skill 1 备用

用法：
```bash
python mix_skill1_skill2.py \
    --skill1 qwen3_cuda_sft_cleaned.jsonl \
    --skill2 skill2_data.jsonl \
    --output mixed_train.jsonl \
    --skill1-n 4500 \
    --report --save-fixed-skill1
```

---

## 九、CoT 全量生成（generate_coder_cot.py）

### 9.1 基于 verdict.decision（rewrite / patch / stop）的 CoT 规则

| decision | CoT 结构 | revised_code 来源 |
|---|---|---|
| `rewrite` | 三段：状态 + top-2 must_change + 约束 | `optimized_code` |
| `patch` | 三段：状态 + top-2 must_change + 约束 | `optimized_code` |
| `stop` | 固定三句（不改代码） | `candidate_code` 原文 |

stop 的 CoT 固定模板：
```
The candidate achieves {speedup:.2f}x speedup with correct results, diagnosed as {diagnosis}. The Verifier decided stop.
No code changes are required. The current implementation is accepted as-is.
Output the candidate code unchanged.
```

### 9.2 全量运行结果

- 生成量：全部 3621 条，100% 覆盖
- 词数：min=31（stop），p50=82，p90=92，max=112
- 越界（< 20 或 > 160 词）：0/3621（0.0%）
- stop 的 `revised_code == candidate_code`：289/289 验证通过
- 原始字段全部完整保留

产出文件：`joint_master_records_with_cot.jsonl`

---

## 十、关键文件状态

| 文件 | 状态 | 说明 |
|---|---|---|
| `relabel_decision_rules.py` | ✓ v2 | 含 Rule 5 perf_type 修正 |
| `joint_master_records.with_7decision.jsonl` | ✓ | 7 值 decision + Rule 5 perf_type |
| `decision_distribution_report.md` | ✓ | 分布报告含 Rule 5 统计 |
| `merged_cards_v2.jsonl` | ✓ | 319 条，新增 7 张 pattern 卡 |
| `append_new_cards.py` | ✓ | 7 张新卡追加脚本 |
| `skill2_data_schema.md` | ✓ | Skill 2 完整 schema 定稿 |
| `mix_skill1_skill2.py` | ✓ | 格式修复 + 混合脚本 |
| `joint_master_records_with_cot.jsonl` | ✓ | 全量 CoT 生成（3621 条） |
| `generate_coder_cot.py` | ✓ | CoT 生成脚本 |

---

## 十一、下一步行动

### 立即可做
1. **Skill 2 JSONL 构建**：基于 schema 和 `joint_master_records_with_cot.jsonl`，组装成带完整 messages 格式的训练文件
2. **feedback_text 补全（613 条）**：模板拼接，无需 Teacher，字段已全部有值
3. **rag_retrieval 补全（902 条）**：对这 902 条跑一遍 RAG 检索，用 `filtered_chunks.jsonl` 补入

### 需要规划
4. **mix_skill1_skill2.py 实际运行**：等 Skill 2 JSONL 就绪后，用 `--skill1-n 4500` 跑混合
5. **Verifier CoT 合成**：设计 teacher prompt，先跑 50 条 pilot 验证质量

### 待设计
6. **Phase 3 闭环采样**：`descriptor_or_workspace_rebuild` 和 `redundant_memory_ops` 的根本解法，需要在采样时有意生成这两类 candidate
7. **Rollback 训练数据**：100-150 条合成数据，从 1676 条合格原材料里构造


## 十一、晚间补充：RAG、Decision 对齐与 Skill 2 导出

### 11.1 official_doc chunking 流程进一步澄清

围绕 `official_doc` 的来源与切块方式，进一步确认了当前仓库里的真实状态：

- `official_doc` 的设计目标不是做通用 RAG 文本切片，而是把官方知识切成 **可转 evidence card 的证据单元**
- chunking 原则来自 [official_doc_schema.md](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc/official_doc_schema.md)：
  - 先把 PDF / HTML 转成纯文本或 markdown
  - 按 section / subsection 切块
  - 单块长度控制在约 `500–1500` 词
  - 先输出 `raw_chunks.jsonl`，再做关键词过滤得到 `filtered_chunks.jsonl`
- 当前仓库里 **保留了 chunk 产物与后续卡片生成脚本**，但没有真正提交 `chunk_docs.py` / `filter_chunks.py` 这两个早期规划中的切块脚本
- 现有 live 产物为：
  - [filtered_chunks.jsonl](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc/filtered_chunks.jsonl)
  - [generate_official_doc_cards.py](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc/generate_official_doc_cards.py)
- 当前 `filtered_chunks` 已扩展到 `343` 条，说明这套 `official_doc` 已从最初的种子规模发展为更宽覆盖版本

### 11.2 `kernel_pattern_db` 与 `benchmark_rule` 的边界对外说明更清晰

基于 [merged_cards_v1.jsonl](/Users/xuwanqi/Desktop/cudaRL/rag/merged_cards_v1.jsonl) 和历史 note，重新整理了两类 source_type 的定义：

- `kernel_pattern_db`
  - 定位：**可迁移的 kernel 问题模式库**
  - 回答的问题：这段代码/性能表现“像不像某种经典 bottleneck 或实现错误”
  - 典型主题：
    - `wrapper_or_launch_overhead`
    - `descriptor_or_workspace_rebuild`
    - `unfused_post_ops`
    - `bad_parallelization`
- `benchmark_rule`
  - 定位：**benchmark 证据是否可信的规则库**
  - 回答的问题：当前 slowdown / speedup 结论是否足够可靠，还是只是 benchmark protocol 有问题
  - 典型主题：
    - missing warmup / steady-state timing
    - missing synchronization
    - apples-to-apples comparison 不成立

对外口径可以统一成：
- `official_doc`：官方约束与硬规则
- `kernel_pattern_db`：典型错误/瓶颈模式
- `benchmark_rule`：性能证据是否可信

### 11.3 用 `merged_cards_v2` 回填 joint_master 缺失 RAG，并顺手修正检索逻辑

针对 [joint_master_records.with_7decision.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_master_verdict/joint_master_records.with_7decision.jsonl) 做了新一轮 RAG 回填，卡源改用 [merged_cards_v2.jsonl](/Users/xuwanqi/Desktop/cudaRL/rag/merged_cards_v2.jsonl)。

初始检查：
- 总记录数：`3621`
- `rag_retrieval` 为空：`241`
- 空缺主要集中在：
  - `correctness_error / semantic_mismatch`：`211`
  - `compile_error / extension_build_failure`：`29`
  - `performance_bottleneck / algorithm_selection_overhead`：`1`

发现旧补全逻辑的问题：
- 主要依赖启发式 subtype 推断，容易把 `semantic_mismatch` 错配到 `shape_contract_mismatch` / `dtype_contract_mismatch` 的泛官方卡
- 对 compile/correctness 的空缺，原逻辑对已有 `verdict.failure_type / perf_type` 利用不充分

因此修改了 [fill_missing_rag_for_joint_master_verdict.py](/Users/xuwanqi/Desktop/cudaRL/datasets/fill_missing_rag_for_joint_master_verdict.py)：
- 新增“**优先按现有 verdict 标签精确检索**”分支
- 有明确 `failure_type / perf_type` 时，先在对应 subtype 卡集合里打分检索
- 只有找不到 subtype 卡时，才回退到旧的启发式 `retrieve_cards()`
- 审计输出里新增：
  - `target_subtype`
  - `retrieval_mode`（`verdict_label_match` / `heuristic_fallback`）

最终回填结果：
- 原空缺 `241` 条全部补齐
- 与备份逐条比对后确认：`filled_from_empty = 241`
- 新命中 top cards：
  - `kernel_pattern_db, correctness error patterns v1 — index mapping / coordinate transform`：`211`
  - `kernel_pattern_db, correctness error patterns v1 — reduction synchronization bug`：`131`
  - `PyTorch Documentation: CUDAExtension and Compiler Flags`：`29`
  - `kernel_pattern_db, compile error patterns v1 — undeclared symbol / API signature`：`17`

辅助产物：
- 审计：[joint_master_records.with_7decision.rag_v2_fill_audit.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_master_verdict/joint_master_records.with_7decision.rag_v2_fill_audit.jsonl)
- 汇总：[joint_master_records.with_7decision.rag_v2_fill_summary.json](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_master_verdict/joint_master_records.with_7decision.rag_v2_fill_summary.json)
- 备份：[joint_master_records.with_7decision.jsonl.bak](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_master_verdict/joint_master_records.with_7decision.jsonl.bak)

### 11.4 按 workflow 把 joint_master 的 decision 对齐到 3-way 空间

发现当前 workflow 主文档 [current_workflow_summary.md](/Users/xuwanqi/Desktop/cudaRL/current_workflow_summary.md) 的顶层要求已经收敛为：

- `decision ∈ {patch, rewrite, stop}`

而现有 joint master 中仍保留 earlier 7-way 标签：
- `local_patch`
- `rewrite_kernel`
- `revise_host`
- `switch_strategy`
- `stop`

因此新增脚本 [align_joint_master_decisions_to_workflow.py](/Users/xuwanqi/Desktop/cudaRL/datasets/align_joint_master_decisions_to_workflow.py)，执行了保守映射：

- `local_patch -> patch`
- `rewrite_kernel / revise_host / switch_strategy / rollback / replan -> rewrite`
- `stop -> stop`

并保留原始细粒度标签：
- 新增 `verdict.decision_7way`
- `verdict.decision` 改写为 3-way workflow 版本

对齐结果：
- `rewrite`: `2912`
- `patch`: `420`
- `stop`: `289`

相关文件：
- 主文件：[joint_master_records.with_7decision.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_master_verdict/joint_master_records.with_7decision.jsonl)
- 备份：[joint_master_records.with_7decision.jsonl.decision7.bak](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_master_verdict/joint_master_records.with_7decision.jsonl.decision7.bak)
- 汇总：[joint_master_records.with_7decision.decision_alignment_summary.json](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_master_verdict/joint_master_records.with_7decision.decision_alignment_summary.json)

### 11.5 从 joint_master_records_with_cot 导出 decision 检查样本

为人工检查 `patch / rewrite / stop` 三种 decision 的完整样本，新增脚本：
- [export_joint_master_decision_check_examples.py](/Users/xuwanqi/Desktop/cudaRL/datasets/export_joint_master_decision_check_examples.py)

从 [joint_master_records_with_cot.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/joint_master_records_with_cot.jsonl) 中各抽取 1 条完整原始 JSON，生成：
- [check_examples.md](/Users/xuwanqi/Desktop/cudaRL/datasets/check_examples.md)

当前抽样到的三条样本：
- `patch`：`joint_l2_100_100_convtranspose3d_clamp_min_divide_pair_corr_781775b6`
- `rewrite`：`joint_l1_98_98_kldivloss_pair_perf_40367774`
- `stop`：`joint_l2_56_56_matmul_sigmoid_sum_pair_stag_e61cd63e`

补充确认：这份 `with_cot` 数据的 CoT 主要落在 `coder_target.coder_cot`，不是单独的 `verdict.cot` 字段。

### 11.6 将 joint_master_records_with_cot 导出成 skill2_schema.md 要求的训练格式

根据 [skill2_schema.md](/Users/xuwanqi/Desktop/cudaRL/skill2_schema.md) 新增脚本：
- [build_skill2_data_from_joint_master.py](/Users/xuwanqi/Desktop/cudaRL/datasets/build_skill2_data_from_joint_master.py)

把 [joint_master_records_with_cot.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/joint_master_records_with_cot.jsonl) 全量重组为标准 Skill 2 训练格式，生成：
- [skill2_data.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/skill2_data.jsonl)

格式特征：
- `messages = [system, user, assistant]`
- `system` 统一：`You are a CUDA kernel optimization expert.`
- `user` 按 schema 组织：
  - Task
  - Reference PyTorch Code
  - Task Contract
  - Current Implementation
  - Execution Feedback
  - Verifier Verdict
  - Requirements
- `assistant` 组织为：
  - `## Thought`
  - `coder_target.coder_cot`
  - `## Revised Code`
  - fenced `python` code block（`coder_target.revised_code`）

全量导出结果：
- 总数：`3621`
- `rewrite`: `2912`
- `patch`: `420`
- `stop`: `289`
- `trainable_for_coder=True`: `3332`
- `trainable_for_coder=False`: `289`

### 11.7 对 skill1 / skill2 / mixed_train 的进一步训练判断

#### （1）关于 Skill 2 是否应显式引入 CoT / enable_thinking

结合 workflow 与当日讨论，最终建议进一步明确为：

- `skill2` 比 `skill1` 更值得引入 reasoning，因为它本质上是 **feedback-to-patch** 任务
- 但不建议把“加 reasoning”直接等价成“全量长 CoT + `enable_thinking=true`”
- 更稳的方案是：
  1. 先从 `skill1` checkpoint 继续训一版 `skill2` 的 code-only patch 版本
  2. 再在输入侧加入短 checklist / patch planning
  3. 最后把 `enable_thinking=true` 当成小剂量 A/B 实验，而不是主线默认配置

对应判断：
- 主线更适合：`enable_thinking=False + 显式结构化 ## Thought/Checklist` 或 checklist 输入化
- `enable_thinking=True` 只建议在 hard subset 上小规模验证

#### （2）对 `mixed_train.jsonl` 是否适合直接用于 Coder v2 训练的判断

检查了 [mixed_train.jsonl](/Users/xuwanqi/Desktop/cudaRL/mixed_train.jsonl) 的实际组成：

- 总数：`8121`
- `skill1`: `4500`
- `skill2`: `3621`
- 所有样本都是合法的 `system / user / assistant` 三段式
- `system` 全部统一
- `skill1` assistant 以 `## Code` 开头
- `skill2` assistant 以 `## Thought` 开头

进一步发现：
- `skill2` 中 `trainable_for_coder=False` 的 `stop` 样本有 `289` 条
- 长样本主要集中在 `skill2`：
  - 约 `2887` 条粗估超过 `4096` token
  - 约 `170` 条超过 `8192` token

最终结论：
- **可以用于 Coder v2 的 mixed SFT 冷启动训练**
- 但不建议不做筛选就直接当最终主训练集硬喂

风险点：
1. 它会把 `skill2` 的显式 `## Thought` 训进 assistant 输出
2. 包含 `stop` / `trainable_for_coder=False` 样本，若训练脚本不读该字段会一起参与训练
3. `skill2` 序列偏长，4k/8k 上限下会有大量截断
4. `skill1` 与 `skill2` 输出协议不统一（`## Code` vs `## Thought + ## Revised Code`）

建议最少做两个派生版本：
- `trainable_only`：过滤 `trainable_for_coder=False`
- `trainable_no_cot`：在保留 Skill 2 信息的前提下去掉 assistant 显式 CoT，维持 code-only 输出协议

---

## 十二维持结论

到今天结束，晚间新增工作主要把白天已经成型的 joint master / CoT / skill2 方案继续向“**可直接训练、可直接抽检、可直接对齐 workflow**”推进了一步：

- RAG 侧：用 `merged_cards_v2` 把剩余空缺 RAG 全部补齐，并修了检索逻辑
- 决策侧：把 joint master 顶层 decision 对齐到 workflow 的 `patch / rewrite / stop`
- 数据侧：把 `joint_master_records_with_cot` 变成了可直接给 Skill 2 用的标准 `skill2_data.jsonl`
- 训练侧：明确了 `mixed_train.jsonl` “能训但不够干净”的边界条件，以及下一步应优先做 trainable-only / no-cot 派生版本

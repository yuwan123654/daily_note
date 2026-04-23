# Daily Note — 2026-04-23

## 今日工作概览

今天围绕四条主线推进：**Verifier SFT schema 与训练策略定稿**、**Skill 1/2 训练数据工程收尾**、**Verifier 标签体系与原始字段映射梳理**、**RAG evidence card 接入 Verifier 的检索方案与抽检**。

---

## 一、Verifier SFT Chat Template Schema

### 1.1 Verifier SFT 定位

Verifier SFT 的目标不是生成 CUDA 代码，而是学习从当前执行状态中判断：

- 当前失败或瓶颈属于什么 diagnosis
- 下一步策略应为 `patch` / `rewrite` / `stop`
- 判断依据是什么
- 如果继续优化，`patch_spec` 应约束 Coder 做什么、不做什么、优先修改哪里

与 Coder skill2 不同，Verifier 输出的是结构化 verdict，不应输出 `revised_code`。

### 1.2 推荐 chat template

采用 ShareGPT / messages 格式：

```jsonc
{
  "sample_id": "...",
  "skill": "verifier_sft",
  "sample_type": "...",
  "messages": [
    {
      "role": "system",
      "content": "You are a CUDA optimization verifier and strategy selector..."
    },
    {
      "role": "user",
      "content": "## Task\n...\n\n## Reference Code\n...\n\n## Current CUDA Code\n...\n\n## Execution Feedback\n...\n\n## Available Evidence (RAG)\n..."
    },
    {
      "role": "assistant",
      "content": "## Diagnosis\n...\n\n## Decision\n...\n\n## Reasoning\n...\n\n## Evidence\n...\n\n## Patch Spec\n..."
    }
  ]
}
```

### 1.3 字段映射

来自 `joint_master_records_with_cot.jsonl` 的输入字段：

- `task_name`
- `reference_task_code`
- `task_contract`
- `candidate_code`
- `execution_evidence`
- `verdict.evidence.rag_retrieval`

assistant 目标字段来自：

- `verdict.diagnosis`
- `verdict.decision`
- `verdict.failure_type`
- `verdict.perf_type`
- `verdict.confidence`
- `verdict.evidence.observations`
- `verdict.evidence.rag_retrieval`
- `verdict.patch_spec`

不应放入 Verifier SFT 的字段：

- `coder_target`
- `coder_cot`
- `optimized_code` / `revised_code`
- `decision_7way` 作为主训练输出

当前 workflow 应以三值 decision 为主：`patch` / `rewrite` / `stop`。`decision_7way` 可以保留为 metadata 或后续 DPO 分析字段，但不建议作为 Phase 1 Verifier SFT 的主输出标签。

---

## 二、Reasoning 与 enable_thinking 策略

### 2.1 `## Reasoning` 是否显式输出

结论：**Phase 1 / 2B 建议显式输出短 `## Reasoning`，但不要训练长 CoT。**

推荐形式是 policy-level rationale，解释为什么选择 `patch` / `rewrite` / `stop`，而不是展示完整链式推理。

示例：

```text
## Reasoning
The code is correct but the measured speedup is below the expected range.
The bottleneck is likely launch/wrapper overhead rather than a local arithmetic bug,
so local patching is unlikely to help. A rewrite should restructure the kernel/host boundary.
```

这样能让模型学习：

```text
execution feedback -> diagnosis -> decision -> patch_spec
```

同时避免输出变长、schema 不稳定、模板化 CoT 噪声放大等问题。

### 2.2 enable_thinking 设置

当前建议：

```text
Phase 1:
enable_thinking = false
显式输出短 ## Reasoning

Phase 2B:
仍以 enable_thinking = false 为主
继续训练稳定 schema 和 strategy rationale

后续增强阶段:
再少量引入 enable_thinking = true
thinking 中放更完整策略比较
final answer 仍保持固定 schema
```

核心理由：

- 当前 Verifier 数据规模不足以稳定监督 hidden thinking
- SFT 需要优先保证可解析输出格式
- 低质量长 CoT 会损害泛化，让模型学到解释模板而不是决策边界
- 短 `## Reasoning` 对泛化有帮助，因为它提供 diagnosis 到 decision 的中间依据

---

## 三、Skill 1/2 训练数据工程收尾

### 3.1 mixed_train.jsonl 状态

训练集 `mixed_train.jsonl` 共 8121 条：

- Skill 1：4500 条
- Skill 2：3621 条

已完成 LLaMA-Factory 所需 data-info：

- `dataset_info.mixed_train.json`

### 3.2 Skill2 schema 对齐

根据 `skill2_schema.md`，后续将 Skill2 user 中的 `Execution Feedback` 部分删除，使其只保留 schema 要求字段。

相关产物：

- `mixed_train.jsonl`
- `mixed_train.jsonl.pre_remove_execution_feedback.bak`
- `check_examples.md` / `check_skill2_examples.md`

训练时需要注意：LLaMA-Factory 的 SFT loss 默认只作用在 messages 中的 assistant 输出上；messages 外的字段通常不会被纳入训练，除非 data collator 或 template 显式读取。因此 metadata 字段可以用于追踪与审计，但不能依赖它们被模型学习。

---

## 四、Verifier 标签体系与来源

### 4.1 diagnosis 候选值

`joint_master_records_with_cot.jsonl` 中 `diagnosis` 实际有 4 类：

| diagnosis | 数量 | 含义 |
|---|---:|---|
| `performance_bottleneck` | 2174 | 正确性通过，但性能瓶颈明显 |
| `correctness_error` | 797 | 进入正确性检查但结果不对 |
| `compile_error` | 361 | 编译、构建、binding、load extension 失败 |
| `stagnation` | 289 | 当前证据不足以继续有效 patch，或应停止 |

### 4.2 failure_type 候选值

`failure_type` 只在 `compile_error` / `correctness_error` 下非空。

实际出现：

| failure_type | 数量 |
|---|---:|
| `semantic_mismatch` | 762 |
| `extension_build_failure` | 302 |
| `binding_signature_mismatch` | 58 |
| `shape_contract_mismatch` | 25 |
| `dtype_contract_mismatch` | 10 |
| `missing_init_args` | 1 |

脚本允许但当前最终数据未出现：

- `resource_or_timeout`
- `insufficient_evidence`

### 4.3 perf_type 候选值

`perf_type` 只在 `performance_bottleneck` 下非空。

实际出现：

| perf_type | 数量 |
|---|---:|
| `bad_parallelization` | 1376 |
| `memory_bound_access` | 445 |
| `wrapper_or_launch_overhead` | 205 |
| `bad_library_path` | 71 |
| `unfused_post_ops` | 38 |
| `small_workload_overhead` | 15 |
| `redundant_memory_ops` | 11 |
| `algorithm_selection_overhead` | 10 |
| `descriptor_or_workspace_rebuild` | 3 |

### 4.4 标签如何得到

标签不是原始 parquet 直接字段，而是通过以下链路得到：

```text
原始执行结果
  -> execution_evidence
  -> sample_type
  -> RAG 检索辅助证据
  -> teacher model 生成 verdict
  -> 脚本校验候选值和一致性
  -> joint_master_records_with_cot.jsonl
```

`diagnosis` 由 `sample_type` 强约束：

```text
compile_failure -> compile_error
correctness_failure -> correctness_error
performance_optimization -> performance_bottleneck
stagnation_or_accept -> stagnation
```

一致性规则：

```text
diagnosis = compile_error / correctness_error
  -> failure_type 必须非空
  -> perf_type 必须为 null

diagnosis = performance_bottleneck
  -> perf_type 必须非空
  -> failure_type 必须为 null

diagnosis = stagnation
  -> failure_type 必须为 null
  -> perf_type 必须为 null
```

---

## 五、execution feedback 字段来源

### 5.1 error_type / error_message 映射

`level_1_2_3_merged.parquet` 原始字段中没有 `error_type` 和 `error_message`，只有：

```text
Error
Correct
CUDA_Runtime
Max_Diff
```

映射关系：

```text
原始 parquet.Error
  -> raw_artifacts.error
  -> execution_evidence.error_message
```

`error_message` 是清洗和截断后的 `Error`：去掉多余空白，最多保留约 300 字符。

`error_type` 不是直接复制字段，而是由 `row_status` 推断；`row_status` 又由 `Error / Correct / CUDA_Runtime` 共同决定。

规则：

| 原始/中间条件 | row_status | error_type |
|---|---|---|
| `Error` 包含 compile/build/nvcc/linker/undefined symbol/load_inline/fatal error 等 | `compile_failed` | `compile_error` |
| `Correct == False` 且不是 compile-like error | `correctness_failed` | `correctness_error` |
| `Correct == True` 且 `CUDA_Runtime` 存在 | `benchmark_ready` | `none` |
| `Error` 存在但不是 compile-like | `runtime_or_other_error` | `runtime_error` |
| `CUDA_Runtime` 缺失且无明确状态 | `runtime_missing` | `runtime_error` |

可汇报结论：

```text
error_message 直接来自原始 Error 字段，经过空白归一化和截断；
error_type 是基于 Error / Correct / CUDA_Runtime 推断出的标准化错误类型。
```

---

## 六、RAG Evidence Card 接入 Verifier

### 6.1 Evidence card 构造与来源

当前 evidence card 主要来自三类来源：

- `official_doc`：CUDA / PyTorch / cuDNN 等官方文档切 chunk 后整理
- `kernel_pattern_db`：根据常见 kernel 性能/正确性模式人工总结的模式库
- `benchmark_rule`：用于判断 benchmark 证据是否可靠的规则卡

每张卡包含：

- `diagnosis_tags`
- `failure_type_tags`
- `perf_type_tags`
- `pattern_description`
- `common_symptoms`
- `snippet` / `snippet_short`
- `patch_hints`
- `retrieval_keywords`

`kernel_pattern_db` 代表代码或 kernel 层面的可修复模式，例如：

- wrapper/launch overhead
- memory_bound_access
- bad_parallelization
- unfused_post_ops

`benchmark_rule` 代表 benchmark 本身的可信度规则，例如：

- missing warmup
- unsynchronized timing
- baseline mismatch
- noisy benchmark

### 6.2 根据 execution feedback 锁定 0-2 张 RAG

推荐流程：

```text
execution feedback
  -> infer diagnosis
  -> infer subtype
  -> filter cards by diagnosis_tags
  -> rank by subtype + keyword overlap
  -> keep top 0-2
```

先用 `execution_evidence.stage / status / error_type / speedup / max_abs_err` 推断一级 diagnosis：

| execution feedback 信号 | diagnosis |
|---|---|
| `error_type = compile_error` | `compile_error` |
| `error_type = correctness_error` | `correctness_error` |
| correctness passed，但 speedup 明显差 | `performance_bottleneck` |
| benchmark 证据不可靠、反复无提升、原因不清 | `stagnation` |

再用 `error_message / observed_symptoms / candidate_code / ncu_profile / torch_profile` 推 subtype。

检索时先用 `diagnosis_tags` 硬过滤，再用 subtype 标签和关键词 overlap 精排：

```text
query = task_name
      + error_type
      + stage/status
      + observed_symptoms
      + error_message
      + candidate_code 特征词
      + ncu_profile / torch_profile 摘要
      + diagnosis
      + subtype
```

最终规则：

- 得分高且 diagnosis 对齐：保留
- subtype 冲突：丢弃
- 只有泛泛相关：丢弃
- `stagnation`：最多保留 1 张，甚至可以为空
- 实在没有合适证据卡：保持 `rag_retrieval=[]`

核心原则：**宁可不给 RAG，也不要给错 RAG**。错误 evidence card 会把 Verifier 带向错误 diagnosis 或错误 patch_spec。

### 6.3 RAG 检索时序

Round 1：

```text
Executor output
  -> executor_to_diagnosis()
  -> RAG 只用 diagnosis_tags 过滤
  -> Verifier 输入
  -> Verifier 输出 diagnosis + subtype + decision
```

Round 2+：

```text
上轮 Verifier diagnosis + failure_type/perf_type
+ 本轮 Executor feedback
  -> diagnosis_tags + subtype_tags 双重过滤
  -> 更精确 RAG 输入
```

这样可以避免 circular dependency：首轮只用 executor 可确定的粗 diagnosis，后续再利用 Verifier 已输出的 subtype 精检索。

---

## 七、抽检与检查文件

今天生成了两类随机抽检文件，供人工检查 schema 与数据质量。

### 7.1 Joint master 样本抽检

输出文件：

- `check_master_examples.md`

从 `joint_master_records_with_cot.jsonl` 随机抽取 5 条完整样本，包含索引表和完整 JSON。抽样覆盖：

- decision：`patch` / `rewrite`
- diagnosis：`compile_error` / `correctness_error` / `performance_bottleneck`

用途：

- 检查 `execution_evidence`
- 检查 `verdict`
- 检查 `rag_retrieval`
- 检查 `coder_target`

### 7.2 Evidence card 样本抽检

输出文件：

- `check_cards_examples.md`

从 `rag/merged_cards_v2.jsonl` 随机抽取 8 张完整 evidence card，包含索引表和完整 JSON。

本次随机抽样刚好全部为 `official_doc` 类型。如需检查来源覆盖，可以后续再按 `source_type` 分层抽样。

---

## 八、今日产出文件

今天直接产生或更新的关键文件：

- `verifier_sft_schema.md`
- `dataset_info.mixed_train.json`
- `mixed_train.jsonl`
- `mixed_train.jsonl.pre_remove_execution_feedback.bak`
- `check_master_examples.md`
- `check_cards_examples.md`
- `check_skill2_examples.md`
- `daily_summary/daily_note_0423.md`

重要输入文件：

- `/Users/xuwanqi/Downloads/daily_note_0423.md`
- `joint_master_records_with_cot.jsonl`
- `datasets/verifier_v1/level_1_2_3_merged.parquet`
- `rag/merged_cards_v2.jsonl`
- `current_workflow_summary.md`
- `skill2_schema.md`

---

## 九、明日建议

1. 固化 Verifier SFT 数据导出脚本，将 `joint_master_records_with_cot.jsonl` 转为最终 `verifier_sft_data.jsonl`。
2. 对 `## Reasoning` 做模板化合成，保持 60-100 词，避免长 CoT。
3. 对 RAG 检索做一次分层抽检，确保 `source_type`、`diagnosis_tags`、`subtype_tags` 覆盖均衡。
4. 对 `stagnation` 样本单独检查，确认 `decision=stop`、`failure_type=null`、`perf_type=null`、`patch_spec.must_change=[]` 的一致性。
5. 用 20-50 条样本跑 Verifier SFT prompt smoke test，检查 assistant 输出是否稳定可解析。

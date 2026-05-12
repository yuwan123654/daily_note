# Daily Note — 2026-05-11

## 范围

本记录覆盖 `daily_note_0507.md` 之后到当前为止形成的关键决策、已完成数据处理、脚本状态和后续计划。重点放在 Phase 1 Verifier SFT 数据集 schema 对齐、stagnation 补充数据、NCU profile 修复，以及 DeepSeek teacher 重新生成主数据集 `reasoning` / `patch spec` 的流程。

---

## 1. Schema 与输入格式最终收敛

后续训练数据统一采用 clean schema，只保留训练所需的 `messages`。

已经明确：

- User 输入中保留原始执行信息，避免提前塞入 `resource_or_timeout` 这类 teacher/verifier 应该自行诊断的抽象标签。
- `error_type` 可以由 `stage/status/error_message` 推断，因此从 clean schema 中移除。
- `stage=benchmark` 可覆盖 correctness 检查后的 benchmark 阶段，不强制拆出单独 `correctness` stage。
- User prompt 的 requirements 需要包含 verdict 候选值和约束规则，保证训练样本看到的约束与 schema 正文一致。
- Verifier 输入应保持和 Coder 输出一致，即保留 Python `load_inline` 包装代码，而不是只提取裸 CUDA。这样更贴近闭环真实分布，也能让 verifier 诊断 wrapper、binding、host path、launch overhead 等问题。

---

## 2. NCU Profile 指标口径

当前 Phase 1 Verifier 使用的 NCU profile 指标统一为 8 项：

```text
Achieved Occupancy
SM Busy
Memory Throughput
L2 Hit Rate
Executed IPC Active
Warp Cycles Per Issued Instruction
Issue Slots Busy
Max Bandwidth
```

重要修正：

- `DRAM Bandwidth Utilization` 改回 `Max Bandwidth`，因为原始数据里对应的是 Max Bandwidth 指标。
- 主数据集已从 `joint_master_records_with_verdict.jsonl` 对照补全 `Max Bandwidth`。
- 补充 stagnation 数据已对照补齐 `Executed IPC Active`，并统一为 `Max Bandwidth`。
- 补充数据中 NCU profile 为 `Not available at this stage` 的 61 条已删除。

当前主数据集状态：

```text
phase1_verifier_sft_final.jsonl: 3621 rows
其中有 NCU profile 的样本: 2336
Not available at this stage: 1285
```

---

## 3. Stagnation 补充数据处理

从 `phase1_supplement.jsonl` 构造了 true stagnation 冷启动补充数据，目标是补足主数据集中缺少的“多轮优化无进展 -> rewrite”样本。

关键规则：

- `stagnation` 不是 accept，也不是单轮性能慢。
- `stagnation` 应表示多轮尝试没有有效提升，当前路线卡住。
- 对应决策主线为：

```text
diagnosis = stagnation
decision = rewrite
failure_type = None
perf_type = None
```

Action spec 对 stagnation 采用固定强约束路线：

```text
patch_goal: abandon the repeated local patch path and rewrite the implementation around a different optimization strategy
must_change:
  - replace the current optimization approach rather than applying another local patch
  - target the repeatedly observed bottleneck class through a different algorithm/library/kernel structure
must_not_change:
  - do not keep applying small local edits to the same implementation path
  - do not change the task contract, output shape, dtype semantics, or correctness tolerance
priority_region:
  - the whole implementation strategy
```

后续判断：

- 这些固定字段可以在 cold-start 阶段直接一步到位。
- `reasoning` 可以后续用 teacher model 补充，但不应让 teacher 随意改变 stagnation 的核心 action spec 约束。

---

## 4. CUDA 包装策略

针对补充数据中的裸 CUDA，已决定并实现 teacher 包装流程：

- 使用 DeepSeek 将裸 CUDA 包装成包含 `load_inline` 的 Python 代码块。
- 如果原始 CUDA 中已经包含 `PYBIND11_MODULE`，则 `load_inline` 不再传 `functions=`，避免 duplicate module binding / duplicate module definition。
- 删除没有显式 CUDA kernel 或 kernel launch 的样本。
- Verifier 训练输入保留 Python wrapper，不再回退成裸 CUDA。

已经编写并迭代：

```text
datasets/verifier_v1/generate_wrapped_python_from_cuda.py
datasets/verifier_v1/cuda_code_to_python_wrapper_prompt.txt
```

---

## 5. Stagnation Reasoning / Patch Spec 生成与清洗

使用 DeepSeek 为 stagnation 补充数据生成 `reasoning` 和 action spec 后，进行了质量检查与清洗。

关键发现：

- DeepSeek 对库路线建议有时过度泛化，尤其会对 elementwise/reduction/loss 类错误建议 cuBLAS/cuDNN。
- 有些 action spec 会建议 `remove load_inline` 或切换到 pure PyTorch-only，这与 Coder 目标分布不一致。

已完成清洗：

- 对 elementwise / reduction / loss 类，禁止 cuBLAS / cuDNN。
- 保留 fused kernel / reduction tree / memory redesign / launch redesign 方向。
- 删除所有 `remove load_inline` / pure PyTorch-only action spec。

当前补充 stagnation 数据状态：

```text
stagnation_data_wrapped_completed.jsonl: 381 rows
已删除 NCU Not available 的 61 rows
```

---

## 6. 主数据集 DeepSeek 重新生成流程

已编写主数据集 teacher 生成脚本和 prompt，用于重新生成主数据集中的：

```text
observations
rag
reasoning
patch_goal
must_change
must_not_change
priority_region
```

固定不变的监督标签：

```text
diagnosis
decision
failure_type
perf_type
confidence
```

相关文件：

```text
datasets/verifier_v1/generate_main_reasoning_and_spec.py
datasets/verifier_v1/main_reasoning_spec_prompt.txt
```

脚本特性：

- 默认输入：`phase1_verifier_sft_final.jsonl`
- 默认输出：`phase1_verifier_sft_final_regenerated.jsonl`
- 支持 `--pilot`
- 支持 `--resume`
- 支持 `--dry-run`
- 支持 DeepSeek API
- 会写入：

```text
datasets/verifier_v1/output_main_reasoning_spec/completed_records.jsonl
datasets/verifier_v1/output_main_reasoning_spec/rejected_records.jsonl
datasets/verifier_v1/output_main_reasoning_spec/generation_log.jsonl
```

---

## 7. Teacher 生成质量门

pilot 检查后，发现 DeepSeek 主要有以下错误模式：

1. GEMM / Linear rewrite 经常只写成 `1D grid -> 2D grid`，这更像 patch，不像真正 rewrite。
2. cumsum / scan 类经常只写 vectorized / coalesced / unroll，忽略 prefix-scan / suffix-scan 依赖结构。
3. 非 GEMM/conv 类会误建议 cuBLAS / cuDNN / tensor core。
4. 有时建议 remove load_inline 或 pure PyTorch-only。
5. 少量格式问题：reasoning 超长/过短、字段名错误、list 过长。

因此脚本中加入了更强的校验规则：

- scan/cumsum rewrite 必须出现 parallel prefix/suffix/segmented scan 相关策略。
- plain GEMM/Linear rewrite 必须出现 tiled GEMM / shared memory tiling / backend/library route / fused epilogue。
- 非 GEMM/conv 任务不得正向建议 library/tensor-core 路线。
- action spec 不得正向要求 remove load_inline 或 pure PyTorch-only。
- stop 样本固定：

```text
patch_goal = No changes required; current implementation meets target quality.
must_change = ["(none)"]
priority_region = ["(none)"]
```

同时修复了两个校验器误杀：

- `Do not use tensor cores` 不应被判定为“建议 tensor cores”。
- `Do not switch to pure PyTorch-only` 不应被判定为“切换 pure PyTorch-only”。

---

## 8. 主数据集当前生成状态

DeepSeek 全量生成过程中出现余额不足和连接错误，因此尚未完成全部 3621 条。

当前最新状态：

```text
phase1_verifier_sft_final_regenerated.jsonl: 2893 rows
completed_records.jsonl: 2893 rows
latest ok: 2893
latest rejected: 329
latest api_error: 399
completed duplicates: 0
```

其中有 154 条原本 rejected，但在修复校验器误杀后可以通过，已恢复并追加回 regenerated 输出。

恢复动作已完成：

```text
recovered_from_rejected: 154
```

备份文件：

```text
phase1_verifier_sft_final_regenerated.jsonl.bak_recover154_20260512_104931
datasets/verifier_v1/output_main_reasoning_spec/completed_records.jsonl.bak_recover154_20260512_104931
datasets/verifier_v1/output_main_reasoning_spec/generation_log.jsonl.bak_recover154_20260512_104931
```

---

## 9. 当前 rejected 的真实结构

修复校验器后，当前最新 rejected 中仍失败的 329 条主要分布为：

```text
268  plain GEMM/Linear rewrite 只写 grid remapping，没有 tiled GEMM/backend/epilogue
24   cumsum/scan 类没有写 parallel prefix/suffix/segmented scan
16   非 GEMM/conv 任务仍正向建议 library/tensor-core
少量 reasoning 超长/过短、字段名错误、列表过长
```

判断：

- 这 329 条大多是 teacher 生成质量问题，不建议直接放行。
- GEMM 类是最大问题，需要更强 prompt 或针对 GEMM 的二次修复 prompt。
- scan/cumsum 类需要显式强调 dependency-preserving scan，而不是只做 memory coalescing。
- 格式类小问题可以通过自动 repair 或轻量重跑解决。

---

## 10. 后续计划

优先级顺序：

1. 处理 `api_error` 的 399 条。
   - 其中一部分是连接错误，一部分是 DeepSeek 余额不足。
   - API 恢复后使用 `--resume` 继续。

2. 针对 329 条 rejected 做二阶段 recovery。
   - GEMM/Linear 类使用专门 prompt，强制输出 tiled GEMM / backend route / fused epilogue。
   - scan/cumsum 类使用专门 prompt，强制输出 prefix/suffix/segmented scan。
   - 禁止 pure PyTorch-only 和 remove load_inline。

3. 对 recovered + rerun 后的 regenerated 数据做最终 QA。
   - 统计 verdict label 分布是否未被改动。
   - 检查 stop 样本 `(none)` 是否固定。
   - 抽查 GEMM / scan / reduction / conv / correctness_error / compile_error。
   - 确认 `optimized_code` / `optimized target` 没有重新泄漏。

4. 形成 Phase 1 训练版本。
   - 主数据：`phase1_verifier_sft_final_regenerated.jsonl`
   - 补充 stagnation：`stagnation_data_wrapped_completed.jsonl`
   - 合并前需要去重、检查 schema、检查 NCU metric 字段一致性。


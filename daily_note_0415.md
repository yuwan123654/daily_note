# Daily Note

日期：`2026-04-15`

## 今日主线

今天的工作主要围绕两条线展开：

1. `official_doc / RAG / evidence card` 生成链路继续收口  
2. `verifier_v1 / neta` 数据集构造与 CUDA 包装脚本落地

---

## 1. official_doc 卡片生成链路

### 1.1 salvage prompt 问题定位

检查了 `official_doc` 第二轮 salvage 生成逻辑，发现 reject 的主因不是知识内容本身，而是 prompt 对 schema 约束不够硬，模型容易出现：

- `failure_type_tags` / `perf_type_tags` 缺失
- `*_tags` 输出成字符串而不是数组
- `stagnation` 情况下忘记显式输出空数组

### 1.2 salvage prompt 修正

更新了：

- [official_doc_card_prompt_salvage.txt](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc/official_doc_card_prompt_salvage.txt)

新增的关键约束：

- `diagnosis_tags / failure_type_tags / perf_type_tags` 必须始终存在
- 三个 `*_tags` 字段必须始终是 JSON array
- 明确了 4 类 diagnosis 的固定字段组合
- 增加了 mini examples，压制格式漂移

### 1.3 official_doc cards 覆盖评估

检查了：

- [official_doc_cards.jsonl](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc/official_doc_cards.jsonl)

当前统计：

- 总卡数：`216`
- `correctness_error`: `156`
- `performance_bottleneck`: `43`
- `compile_error`: `16`
- `stagnation`: `1`

结论：

- 当前库已经可用
- 但覆盖仍偏向 `correctness_error / semantic_mismatch`
- 后续需要定向补：
  - `performance_bottleneck`
  - `stagnation`
  - `cuDNN / cuBLASLt / cuSPARSELt`
  - `benchmark / profiler interpretation`

---

## 2. 手写 seed cards

### 2.1 第一批 seed cards

新增：

- [kernel_pattern_and_benchmark_seed_cards.jsonl](/Users/xuwanqi/Desktop/cudaRL/rag/kernel_pattern_and_benchmark_seed_cards.jsonl)
- [kernel_pattern_and_benchmark_seed_cards.md](/Users/xuwanqi/Desktop/cudaRL/rag/kernel_pattern_and_benchmark_seed_cards.md)

第一批共 `6` 张：

- `kernel_pattern_db`: `3`
- `benchmark_rule`: `3`

覆盖：

- `wrapper_or_launch_overhead`
- `descriptor_or_workspace_rebuild`
- `unfused_post_ops`
- `warmup / steady-state`
- `synchronization discipline`
- `apples-to-apples benchmark discipline`

### 2.2 第二批 kernel_pattern_db

继续补了第二批 `kernel_pattern_db`，当前总计：

- 总卡数：`10`
- `kernel_pattern_db`: `7`
- `benchmark_rule`: `3`

新增 perf pattern：

- `bad_library_path`
- `redundant_memory_ops`
- `bad_parallelization`
- `memory_bound_access`

这些 seed cards 后续可直接用于：

- RAG 补充
- teacher few-shot
- verifier / optimizer 的规则底座

---

## 3. verifier_v1 包装脚本（单条样本）

### 3.1 目标

基于：

- [level_1.parquet](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/level_1.parquet)

设计 teacher，将其中的 `CUDA_Code` 包装成完整 Python 文件。

### 3.2 已落地文件

- Prompt：
  [cuda_code_to_python_wrapper_prompt.txt](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/cuda_code_to_python_wrapper_prompt.txt)
- 脚本：
  [generate_wrapped_python_from_cuda.py](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/generate_wrapped_python_from_cuda.py)
- 辅助模板说明：
  [python_wrapper_template.md](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/python_wrapper_template.md)

### 3.3 核心约束

包装规则收口为：

- 生成完整单文件 Python
- 保留参考 `Model`
- 新增 `ModelNew`
- 不包含 `get_inputs()` / `get_init_inputs()`
- `ModelNew.__init__` 必须和 `Model.__init__` **完全一致**
- `ModelNew.forward` 必须和 `Model.forward` **完全一致**
- 用 `load_inline` 包装 `CUDA_Code`
- 保守包装，不重写任务语义

### 3.4 校验逻辑升级

脚本中加入了 AST 级校验：

- 禁止出现 `get_inputs`
- 禁止出现 `get_init_inputs`
- 检查 `ModelNew.__init__` 与 `Model.__init__` 的参数签名完全一致
- 检查 `ModelNew.forward` 与 `Model.forward` 的参数签名完全一致

---

## 4. parquet 到 Markdown 检查稿

检查了：

- [level_1_wrapped_python.parquet](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/output_wrapped_python/level_1_wrapped_python.parquet)

发现整份数据量较大：

- 行数：`12157`
- 列数：`21`

先生成了一版检查稿：

- [level_1_wrapped_python_inspection.md](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/output_wrapped_python/level_1_wrapped_python_inspection.md)

但后续用户明确要求“原封不动转换”，因此这条线暂时停在检查版，没有继续做全量原样 md 转储。

---

## 5. meta parquet 排序

按用户要求重排了：

- [level_2.parquet](/Users/xuwanqi/Desktop/cudaRL/datasets/meta/level_2.parquet)
- [level_3.parquet](/Users/xuwanqi/Desktop/cudaRL/datasets/meta/level_3.parquet)

规则：

- 同一 `Task_ID` 组内按 `CUDA_Runtime` 升序
- `CUDA_Runtime = null` 排在组尾

产物：

- [level_2_sorted_by_task_cuda_runtime_nulls_last.parquet](/Users/xuwanqi/Desktop/cudaRL/level_2_sorted_by_task_cuda_runtime_nulls_last.parquet)
- [level_3_sorted_by_task_cuda_runtime_nulls_last.parquet](/Users/xuwanqi/Desktop/cudaRL/level_3_sorted_by_task_cuda_runtime_nulls_last.parquet)

并完成校验：

- 组内非空值升序：通过
- 组内空值最后：通过

---

## 6. neta 三份数据集构造优化前后 pairs

### 6.1 输入数据

基于：

- [level_1.parquet](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/level_1.parquet)
- [level_2.parquet](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/level_2.parquet)
- [level_3.parquet](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/level_3.parquet)

### 6.2 构造目标

按用户指定比例构造“优化前 / 优化后”数据对：

- `compile_error`: `10%`
- `correctness_error`: `25%`
- `performance_bottleneck`: `55%`
- `stagnation`: `10%`

并要求：

- 同一任务组内配对
- `CUDA_Runtime` 尽可能接近
- 尽量避免优化步幅过大

### 6.3 已落地脚本

- [build_optimization_pairs.py](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/build_optimization_pairs.py)

### 6.4 配对逻辑

任务组键：

- `Level_ID + Task_ID + Op_Name`

四类定义：

- `compile_error`
  - before: compile-like error
  - after: 能进入有 `CUDA_Runtime` 的可执行样本
- `correctness_error`
  - before: `Correct=False` 且非 compile error
  - after: `Correct=True`
- `performance_bottleneck`
  - before: `Correct=True` 且 `speedup <= 1.0`
  - after: `Correct=True` 且 runtime 更小、speedup 更高
- `stagnation`
  - before/after: `Correct=True` 且 `speedup > 2.0`
  - after 不差于 before

并使用：

- 组内 runtime 邻近优先
- 样本全局不复用
- 按真实可选容量反推最大可行平衡规模

### 6.5 当前产物

- [optimization_pairs.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/output_pairs/optimization_pairs.jsonl)
- [optimization_pairs.parquet](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/output_pairs/optimization_pairs.parquet)
- [optimization_pairs_summary.json](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/output_pairs/optimization_pairs_summary.json)

### 6.6 当前结果

最终可行的最大平衡数据集：

- 总 pair 数：`904`

分布：

- `compile_error`: `90`
- `correctness_error`: `226`
- `performance_bottleneck`: `497`
- `stagnation`: `91`

和目标比例基本一致：

- `10% / 25% / 55% / 10%`

限制瓶颈：

- `correctness_error` 的真实可选容量只有 `226`
- 因此整套平衡数据集上限被它卡在 `904` 对

### 6.7 抽样检查稿

按类别分层随机抽样，生成：

- [optimization_pairs_sample.md](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/output_pairs/optimization_pairs_sample.md)

抽样分布：

- `compile_error`: `3`
- `correctness_error`: `3`
- `performance_bottleneck`: `4`
- `stagnation`: `2`

用于人工 spot check 配对质量。

---

## 7. 这份 pairs 对 verifier_v1 的判断

基于：

- [verifier_v1_sft_schema.md](/Users/xuwanqi/Desktop/cudaRL/verifier_v1_sft_schema.md)

对当前 `904` 对 pairs 做了判断：

- **足够**作为 `Verifier-v1-alpha / bootstrapping SFT`
- **不够**作为正式版 `Verifier-v1` 的最终训练集

原因：

- schema 需要学的不只是 4 分类，还包括：
  - `failure_type / perf_type`
  - `evidence`
  - `patch_spec`
- 当前 pairs 更像高质量“原料”
- 真正薄的类别：
  - `compile_error`
  - `stagnation`
  - correctness 细粒度 failure_type

结论：

- 这 `904` 对足够启动第一轮 teacher 标注和 alpha 训练
- 后续仍需扩展到几千条，尤其补薄弱类别

---

## 8. pair 级 CUDA 包装脚本

### 8.1 目标

在 `optimization_pairs` 的基础上，对 before / after 的 `CUDA_Code` 都包装成完整 Python。

### 8.2 已落地文件

- Prompt：
  [cuda_pair_to_python_wrapper_prompt.txt](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/cuda_pair_to_python_wrapper_prompt.txt)
- 脚本：
  [generate_wrapped_python_from_pairs.py](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/generate_wrapped_python_from_pairs.py)

### 8.3 设计要点

每个 pair 一次性生成两份完整 Python：

- `before_wrapped_python_code`
- `after_wrapped_python_code`

并保持与单条样本版包装脚本同一套规范：

- 保留 `Model`
- 新增 `ModelNew`
- 不包含 `get_inputs()` / `get_init_inputs()`
- `ModelNew.__init__` 必须与 `Model.__init__` 完全一致
- `ModelNew.forward` 必须与 `Model.forward` 完全一致
- 使用 `load_inline`
- 保守包装，不改任务语义

脚本已通过语法检查。

---

## 今日产出文件清单

- [official_doc_card_prompt_salvage.txt](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc/official_doc_card_prompt_salvage.txt)
- [kernel_pattern_and_benchmark_seed_cards.jsonl](/Users/xuwanqi/Desktop/cudaRL/rag/kernel_pattern_and_benchmark_seed_cards.jsonl)
- [kernel_pattern_and_benchmark_seed_cards.md](/Users/xuwanqi/Desktop/cudaRL/rag/kernel_pattern_and_benchmark_seed_cards.md)
- [cuda_code_to_python_wrapper_prompt.txt](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/cuda_code_to_python_wrapper_prompt.txt)
- [generate_wrapped_python_from_cuda.py](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/generate_wrapped_python_from_cuda.py)
- [python_wrapper_template.md](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/python_wrapper_template.md)
- [level_2_sorted_by_task_cuda_runtime_nulls_last.parquet](/Users/xuwanqi/Desktop/cudaRL/level_2_sorted_by_task_cuda_runtime_nulls_last.parquet)
- [level_3_sorted_by_task_cuda_runtime_nulls_last.parquet](/Users/xuwanqi/Desktop/cudaRL/level_3_sorted_by_task_cuda_runtime_nulls_last.parquet)
- [build_optimization_pairs.py](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/build_optimization_pairs.py)
- [optimization_pairs.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/output_pairs/optimization_pairs.jsonl)
- [optimization_pairs.parquet](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/output_pairs/optimization_pairs.parquet)
- [optimization_pairs_summary.json](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/output_pairs/optimization_pairs_summary.json)
- [optimization_pairs_sample.md](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/output_pairs/optimization_pairs_sample.md)
- [cuda_pair_to_python_wrapper_prompt.txt](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/cuda_pair_to_python_wrapper_prompt.txt)
- [generate_wrapped_python_from_pairs.py](/Users/xuwanqi/Desktop/cudaRL/datasets/verifier_v1/neta/generate_wrapped_python_from_pairs.py)

---

## 下一步建议

1. 先跑一小批 `generate_wrapped_python_from_pairs.py`，验证 before/after 包装质量。  
2. 对 `optimization_pairs_sample.md` 做人工 spot check，确认四类配对是否符合预期。  
3. 在 `904` 对基础上继续定向补 `compile_error / stagnation / correctness 细分`。  
4. 再把 pair 数据 teacher 蒸馏成 `verifier_v1_sft_schema` 的标准样本。  

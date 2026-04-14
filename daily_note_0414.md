# Daily Note

日期：2026-04-14

## 今日完成事项

- 综合阅读并吸收了外部笔记：
  - [daily_note_2026-04-14.md](/Users/xuwanqi/Downloads/daily_note_2026-04-14.md)
- 将外部笔记中的 Verifier-RAG 设计思路，与今天项目内实际推进的工作进行了统一整理。
- 明确了今天的主线聚焦在：
  - `Verifier-v1 schema`
  - `RAG` 在 CUDA 算子优化场景中的职责
  - `official_doc` 官方知识库建设
  - `evidence card` 的种子数据构造

## 一、Schema 与 RAG 目标进一步对齐

今天继续确认了当前 `Verifier-v1` 的基本方向：

- 顶层采用单一 `verdict`
- `diagnosis` 固定为 4 类：
  - `compile_error`
  - `correctness_error`
  - `performance_bottleneck`
  - `stagnation`
- 在顶层判断之下，细分：
  - `failure_type`
  - `perf_type`
- `evidence` 同时包含：
  - `observations`
  - `rag_retrieval`

进一步明确：

- `Verifier` 不是普通分类器
- 它是一个**面向 patch 的统一判断器**
- `RAG` 的作用不是替代模型做结论，而是为：
  - `diagnosis`
  - `failure_type / perf_type`
  - `why_relevant`
  - `patch_spec`
  提供外部证据支持

今天也再次确认：

- **RAG 不负责下结论**
- **Verifier 才负责下结论**

这意味着：

- RAG 负责检索：
  - 官方规则
  - 经验模式
  - benchmark 边界
  - 库级约束
- Verifier 负责结合：
  - reference / candidate
  - execution evidence
  - retrieved evidence
  输出最终结构化 `verdict`

## 二、RAG 的知识组织方式进一步明确

### 1. `kernel_pattern_db` 的定位

今天确认：

- `kernel_pattern_db` 不是现成工具名
- 更适合定义为：
  - 面向 CUDA / PyTorch 算子修错与优化的**模式知识库**

它与其他来源的边界更清楚了：

- `official_doc`
  - 官方规则与硬约束
- `project_history`
  - 项目历史案例
- `benchmark_note`
  - benchmark 经验与边界
- `kernel_pattern_db`
  - 可迁移的错误模式 / 性能瓶颈模式

### 2. 最小知识单元应为 evidence card

今天继续强化的共识是：

- 不应直接把长文档整体作为检索目标
- Verifier 真正需要的是：
  - 一条短的
  - 可归因的
  - 能直接支撑 `verdict` 的证据

因此最小知识单元应该是：

- **evidence card**

并且一张卡尽量只描述一个主问题，不混合：

- shape 错
- semantic 错
- perf 慢点

### 3. Query 应该是结构化信号

今天继续确认：

- verifier-RAG 不应主要依赖自然语言问句
- query 更适合由样本中的结构化信号构成

例如：

- `diagnosis_route_hint`
- `operator_info`
- `execution_info`
- `metrics`
- `contract_signals`
- `observed_symptoms`
- `code_signals`
- `log_signals`
- `hard_filters`

这使得 query 可以更好地和 evidence card 中的：

- `trigger_signals`
- `code_patterns`
- `log_patterns`
- `patch_hints`

对齐。

## 三、official_doc 库建设：今天完成了两轮补库

### 1. 补库目标

今天判断当前 `official_doc` 已经可用，但还不够“覆盖全面”。  
重点缺口集中在：

- `cuDNN` 更深的 support surface / execution / layout 约束
- `PTX / async pipeline`
- `cuSPARSELt`

因此今天直接做了两轮 focused 补库，而不是继续只停留在评估层面。

### 2. 实际补进的内容

新增并整理进官方库的重点主题包括：

- `cuDNN`
  - custom execution plan
  - dynamic shapes / kernel cache
  - graph heuristics / fallback
  - support surface / g2 shape / layout restrictions
  - attention layout / head-dim constraints / determinism
  - JIT / forward-compatibility / warmup overhead
- `PTX / async pipeline`
  - libcu++ async operations
  - PTX API version / SM gating
  - `cp.async.bulk`
  - `mbarrier`
  - `TMA`
  - async proxy / generic proxy ordering
  - tensor-core async / WGMMA eligibility
- `cuSPARSELt`
  - descriptor / layout / alignment constraints
  - structured sparsity workflow
  - plan / workspace / search behavior
  - matmul semantics / one-sparse-operand rule
  - prune algorithm / validation / preprocess amortization

### 3. 官方库当前状态

当前 live 文件：

- [filtered_chunks.jsonl](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc/filtered_chunks.jsonl)

当前统计：

- 总 chunk 数：`343`
- 主要分布：
  - `CUDA Programming Guide`: `197`
  - `CUDA Runtime API`: `22`
  - `cuBLAS Library`: `22`
  - `CUDA C++ Best Practices Guide`: `21`
  - `cuDNN family`: `19`
  - `PTX / async family`: `8`
  - `cuSPARSELt family`: `5`
  - `CUB family`: `4`

### 4. 覆盖度重新评估

今天在补库后重新给出覆盖度判断：

- 作为 CUDA 算子优化检索库：`8.8/10`
- 作为“覆盖全面的 official_doc 库”：`8/10`

结论：

- 现在已经足够支撑 `Verifier / Optimizer` 的正式第一版 RAG
- 已经不只是实验库
- 但还存在进一步加厚空间，主要是：
  - `CUDA Programming Guide` 占比仍然偏高
  - `CUB` 仍偏薄
  - 更底层的 tensor-core / async barrier / specialized PTX 还可继续补

### 5. 今日更新的补库脚本与摘要

- 补库脚本：
  - [append_gap_chunks.py](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc/append_gap_chunks.py)
- 来源摘要：
  - [rag_source_summary.md](/Users/xuwanqi/Desktop/cudaRL/rag_source_summary.md)

今天也把来源摘要更新到当前 live 库状态，修正了旧版 summary 仍引用旧路径、旧计数的问题。

## 四、从 official_doc 中开始构造第一批 evidence card

今天决定从 `official_doc` 开始手工写第一批 seed cards，原因是：

- 内容相对稳定
- 不依赖项目私有数据
- 很适合先验证：
  - card schema
  - teacher prompt
  - verifier 对 `rag_retrieval` 的消费方式

### 1. 对齐的 schema 文件

- [official_doc_schema.md](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc_schema.md)
- [card_schema.md](/Users/xuwanqi/Desktop/cudaRL/rag/card_schema.md)

### 2. 已手工完成的 seed cards

今天一共手工写了 `6` 张 official-doc seed cards：

- JSONL 版本：
  - [official_doc_seed_cards.jsonl](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc_seed_cards.jsonl)
- Markdown 检查版：
  - [official_doc_seed_cards.md](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc_seed_cards.md)

这 6 张卡覆盖了比较典型的几类问题：

1. `cuDNN Frontend dynamic-shape / kernel cache`  
   - 侧重：
     - `performance_bottleneck`
     - `stagnation`
     - wrapper / plan rebuild 开销

2. `Async proxy / cp.async / TMA ordering`  
   - 侧重：
     - `correctness_error`
     - `semantic_mismatch`

3. `cuSPARSELt plan / workspace / search steady-state overhead`  
   - 侧重：
     - `performance_bottleneck`
     - `stagnation`

4. `cuDNN support surface / g2 shape / layout restrictions`  
   - 侧重：
     - `correctness_error`
     - `shape_contract_mismatch`

5. `cp.async.bulk completion mechanism / mbarrier mismatch`  
   - 侧重：
     - `correctness_error`
     - `performance_bottleneck`

6. `cuSPARSELt workflow semantic misuse`  
   - 侧重：
     - `correctness_error`
     - `stagnation`

这些种子卡的设计原则是：

- `pattern_description` 要描述具体模式，而不是泛概念
- `common_symptoms` 要贴近 verifier 实际看到的现象
- `patch_hints` 要可执行，能直接支撑后续 `patch_spec`

## 五、外部笔记中值得纳入项目主线的结论

今天还吸收了外部笔记里几条很重要的结论，并和项目主线合并：

### 1. RAG 可以且应该批量构造

合理路径不是手写全库，而是：

- 候选卡片批量生成
- 规则质检
- 去重与 canonical 化
- 抽检与回修

推荐并行的四条构造线：

1. 模板驱动批量构造
2. 历史样本蒸馏
3. 代码模式挖掘
4. 官方文档蒸馏

### 2. 没有现成数据也可以先启动 verifier-RAG

不必等：

- project history
- 大规模标注样本
- 完整生产数据

再开始做 RAG。

更好的方式是：

- 先用 `official_doc`
- 再配合 `kernel_pattern_seed`
- 再补少量 `benchmark_note`

先搭一个**最小可运行版 verifier-RAG**。

### 3. official_doc 的范围不应只限于 CUDA 基础原理

今天进一步整合后明确：

`official_doc` 必须覆盖：

- CUDA 基础执行模型
- Runtime API
- cuBLAS / cuBLASLt
- cuDNN
- profiling 与 benchmark 文档
- 版本兼容、support matrix、JIT / forward-compatibility 相关信息

### 4. Torch 版本锚点策略

外部笔记里建议：

- 用稳定版 Torch 作为主锚点版本
- 本地保存文档快照
- 在知识库元数据中记录：
  - `runtime_anchor`
  - `doc_snapshot_date`

这个方向与我们当前“官方库需要可追溯”的需求是一致的。

## 六、今天新增或更新的关键文件

- [append_gap_chunks.py](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc/append_gap_chunks.py)
- [filtered_chunks.jsonl](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc/filtered_chunks.jsonl)
- [rag_source_summary.md](/Users/xuwanqi/Desktop/cudaRL/rag_source_summary.md)
- [official_doc_seed_cards.jsonl](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc_seed_cards.jsonl)
- [official_doc_seed_cards.md](/Users/xuwanqi/Desktop/cudaRL/rag/official_doc_seed_cards.md)

## 七、关键结论

### 1. 官方知识库已经从“可用原型”升级到“可正式用于第一版 Verifier-RAG”

今天补库后的 `official_doc` 已经不只是泛 CUDA 文档集合，而是：

- 更贴近 CUDA 算子优化
- 更适合做 verdict 支撑
- 更适合让 Verifier 生成：
  - `why_relevant`
  - `patch_spec`

### 2. 证据卡片路线已经开始落地

今天不只是讨论了 evidence card，而是已经开始实际写 seed 数据。  
这意味着：

- card schema
- official_doc 蒸馏方式
- future teacher prompt few-shot

已经进入可执行阶段。

### 3. 当前最合理的下一步是打通 official_doc 蒸馏链路

相比继续扩展大而全的文档库，当前更高收益的是：

- 基于现有 `official_doc` 库
- 用这 6 张 seed cards
- 设计并跑通第一版 teacher prompt
- 批量生成 `official_doc` evidence cards

## 八、下一步建议

1. 基于现有 `6` 张 seed cards，写第一版 `official_doc` teacher prompt 模板。  
2. 先从 `official_doc/filtered_chunks.jsonl` 中抽小批量 chunk，跑 teacher 生成候选 cards。  
3. 做规则校验：
   - schema 完整性
   - diagnosis / failure_type / perf_type 一致性
   - patch hints 是否可执行  
4. 对 `official_doc` 类 card 跑首轮去重与抽检。  
5. 在 official-doc 蒸馏链路稳定后，再复用到：
   - `kernel_pattern_db`
   - `benchmark_note`
   - `project_history`

## 一句话总结

今天完成了从 **Verifier-RAG 设计** 到 **official_doc 补库** 再到 **6 张 evidence card seed 手工落地** 的完整推进。  
当前已经具备进入下一阶段的条件：

> 可以正式开始构建 `official_doc -> evidence card -> verifier prompt` 的第一版批量蒸馏链路。

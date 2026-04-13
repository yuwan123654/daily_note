# Daily Note

日期：2026-04-13

## 今日完成事项

- 阅读并对齐了 [daily_note_0410.md](/Users/xuwanqi/Desktop/cudaRL/daily_summary/daily_note_0410.md) 的阶段结论与后续方向。
- 明确当前主线继续沿着：
  - `Verifier-first`
  - `Coder-v1-alpha -> Coder-v1-final`
  - `assistant` 保持 code-only / JSON-only 输出
  - 优先补 `shape / reduction / correctness / perf` 相关能力
- 设计并多轮迭代了 `Verifier-v1` 的 SFT schema：
  - 初版为结构化 Verifier 输出 schema
  - 后续加入“错误修正 + 性能优化”双目标
  - 再简化为单一 `verdict` 结构
  - 最终保留：
    - 顶层 `diagnosis`
    - 子标签 `failure_type / perf_type`
    - 结构化 `patch_spec`
    - `evidence.observations + evidence.rag_retrieval`
- 将 [verifier_v1_sft_schema.md](/Users/xuwanqi/Desktop/cudaRL/verifier_v1_sft_schema.md) 改为中文说明版，同时保持训练目标中的字段和值使用英文。
- 新增了 Verifier 示例文件：
  - [verifier_v1_example_samples.md](/Users/xuwanqi/Desktop/cudaRL/verifier_v1_example_samples.md)
- 讨论并确认了 Verifier 是否需要 thinking：
  - 不加 thinking 会影响复杂歧义样本的泛化
  - 但当前阶段更关键的是 schema 稳定性、标签边界和样本覆盖
  - 因此 `Verifier-v1` 可以先采用“无 thinking、强结构化”的路线
- 针对 `level_1.parquet` 做了数据处理：
  - 先按 `Op_Name`
  - 再按 `CUDA_Runtime` 升序
  - 并生成空值排在组末尾的新 parquet：
    - [level_1_sorted_by_op_name_cuda_runtime_nulls_last.parquet](/Users/xuwanqi/Desktop/cudaRL/level_1_sorted_by_op_name_cuda_runtime_nulls_last.parquet)
- 基于 [level_1.parquet](/Users/xuwanqi/Desktop/cudaRL/level_1.parquet) 设计了使用 teacher model 批量生成 Verifier 训练数据的实施方案。
- 直接实现了第一步预处理脚本：
  - [prepare_intermediate_jsonl.py](/Users/xuwanqi/Desktop/cudaRL/scripts/verifier/prepare_intermediate_jsonl.py)
- 使用该脚本将 [level_1.parquet](/Users/xuwanqi/Desktop/cudaRL/level_1.parquet) 转成 teacher-ready 中间 JSONL：
  - [verifier_level1_intermediate.jsonl](/Users/xuwanqi/Desktop/cudaRL/verifier_level1_intermediate.jsonl)

## 关键结论

### 1. 当前 Verifier schema 的定型方向

当前更合适的 Verifier 结构是：

- 单一 `verdict`
- 顶层 `diagnosis` 固定为 4 类：
  - `compile_error`
  - `correctness_error`
  - `performance_bottleneck`
  - `stagnation`
- 在此基础上细分：
  - `failure_type`
  - `perf_type`
- `evidence` 使用结构化对象：
  - `observations`
  - `rag_retrieval`

这样做的优点是：

- 顶层判断简单
- 子标签足够细
- 既适合修错，也适合性能优化
- 适合直接为后续 DPO 做准备

### 2. 数据占比需要偏向算子优化

因为最终目标是算子优化，而不是一般性故障分类，所以 Verifier 的 SFT 数据分布需要明显偏向性能类样本。

今天确认的单阶段建议配比为：

- `compile_error`: `15%`
- `correctness_error`: `35%`
- `performance_bottleneck`: `40%`
- `stagnation`: `10%`

这版配比的用途是：

- SFT 阶段一步到位
- 不再单独拆训练阶段
- 直接为后续 DPO 构造 chosen/rejected pair 做准备

### 3. RAG 暂时保留占位，不在首轮预处理时填充

虽然 schema 已经支持：

- `evidence.observations`
- `evidence.rag_retrieval`

但为了先打通数据链路，今天决定：

- 首轮预处理脚本中 `rag_context.retrieval_candidates` 先统一输出空数组
- 先完成 parquet -> intermediate JSONL
- 后续再单独接入检索模块

### 4. intermediate JSONL 已经跑通

预处理脚本已成功把 `level_1.parquet` 转成中间 JSONL。

当前输出文件：

- [verifier_level1_intermediate.jsonl](/Users/xuwanqi/Desktop/cudaRL/verifier_level1_intermediate.jsonl)

总样本数：

- `12157`

脚本当前的规则预标注分布为：

- `compile_error = 2341`
- `correctness_error = 2594`
- `performance_bottleneck = 3754`
- `stagnation = 3468`

这里的 `stagnation` 目前只是**粗规则桶**，判定逻辑是：

- `Correct=True`
- 且 `CUDA_Speedup_Native >= 0.95`
- 且没有明显 compile / correctness 失败信号

因此它当前更接近：

- “暂时不优先处理”
- “接近可接受”
- “等待 teacher 再细判”

而不是最终高置信语义标签。

### 5. 当前 `candidate_code` 的真实范围

在当前 intermediate JSONL 中：

- `candidate_code <- CUDA_Code`

因此它并不是完整 Python 候选文件，而是：

- 原始 CUDA / C++ 扩展代码字符串
- 更准确的范围是：
  - `candidate_scope = "cuda_kernel_only"`

这意味着后续 teacher prompt 里必须明确说明：

- 当前候选实现只覆盖 kernel / extension 侧
- 不一定含完整 wrapper / `ModelNew`

## 今日新增或更新的关键文件

- [verifier_v1_sft_schema.md](/Users/xuwanqi/Desktop/cudaRL/verifier_v1_sft_schema.md)
- [verifier_v1_example_samples.md](/Users/xuwanqi/Desktop/cudaRL/verifier_v1_example_samples.md)
- [prepare_intermediate_jsonl.py](/Users/xuwanqi/Desktop/cudaRL/scripts/verifier/prepare_intermediate_jsonl.py)
- [verifier_level1_intermediate.jsonl](/Users/xuwanqi/Desktop/cudaRL/verifier_level1_intermediate.jsonl)
- [level_1_sorted_by_op_name_cuda_runtime_nulls_last.parquet](/Users/xuwanqi/Desktop/cudaRL/level_1_sorted_by_op_name_cuda_runtime_nulls_last.parquet)

## 下一步建议

1. 基于 [verifier_level1_intermediate.jsonl](/Users/xuwanqi/Desktop/cudaRL/verifier_level1_intermediate.jsonl) 生成 teacher prompt JSONL。  
2. 在 teacher prompt 中明确：
   - `candidate_code` 是 `cuda_kernel_only`
   - 顶层只允许一个 `verdict`
3. 先小规模跑 `200-300` 条 teacher 样本，检查：
   - `diagnosis`
   - `failure_type / perf_type`
   - `patch_spec`
   - `evidence` 写法
4. 确认 teacher 质量后，再批量扩展到首轮可用规模。  
5. 后续再单独接入 RAG 检索模块，把 `rag_retrieval` 从空数组升级成真实证据。  

## 补充：

这份笔记重点围绕 **Verifier-v1 schema 对应的 RAG 构造方案**，补充了今天主线里还没完全落细的部分，尤其是：

- `kernel_pattern_db` 的定位
- evidence card 的最小知识单元设计
- query schema 的结构化设计
- 在当前“几乎没有现成数据”的情况下，如何从 0 启动 verifier-RAG

### 1. RAG 在 Verifier 中的职责边界

今天进一步明确：

- **RAG 不负责下结论**
- **Verifier 才负责下结论**

也就是说：

- RAG 负责：
  - 找证据
  - 补知识
  - 提供经验模式
  - 提供官方约束
- Verifier 负责：
  - 结合 reference / candidate / execution evidence / retrieved evidence
  - 输出最终 `verdict` JSON

这和当前 schema 完全一致，因为检索结果最终应该进入：

- `verdict.evidence.rag_retrieval`

### 2. `kernel_pattern_db` 的正式定位

今天澄清了一个重要概念：

- `kernel_pattern_db` 不是现成工具，也不是固定术语

它更适合被定义为：

- 面向 CUDA / PyTorch 算子修错与优化的**模式知识库**

它的作用不是存普通文本，而是把常见模式整理成可检索的经验卡，例如：

- 常见错误模式
- 常见性能瓶颈模式
- 常见 patch 区域
- 常见必须保持不变的约束

### 3. RAG 的最小知识单元应该是 evidence card

今天形成的关键共识是：

- 不应该直接检索整篇文档
- Verifier 真正需要的是“一条能直接支撑 verdict 的短证据”

因此，最小知识单元应该是：

- **evidence card**

而不是普通长文档切块。

这类卡片应尽量只表达一个主模式，不要把：

- shape 错
- semantic 错
- perf 慢点

混在一张卡里。

### 4. evidence card 需要内置 patch 线索

因为 Verifier 不是只分类，还要输出 `patch_spec`，所以 evidence card 最好天然带有 patch 线索。

今天确认适合补进卡片的内容包括：

- `patch_goal_template`
- `must_change_hints`
- `must_not_change_hints`
- `priority_region_hints`

这样 teacher 或 Verifier 在生成最终 `patch_spec` 时更容易输出可执行结果，而不是空泛建议。

### 5. query 应该是结构化信号，不是自然语言问题

今天明确：

- verifier-RAG 场景里的 query 不应主要依赖自然语言提问
- 更适合的是从当前样本抽取出来的一组结构化信号

这些 query 信号建议至少覆盖：

- stage
- status
- error_type
- metrics
- observed_symptoms
- code_signals
- log_signals
- diagnosis_route_hint

这和后续你要做的 teacher prompt / feature extractor 是直接对应的。

### 6. 证据卡片可以批量构造，但不能一步全自动入库

今天确认的最佳策略是：

- **候选卡片批量生成 + 规则校验 + 去重聚类 + 人工抽检回修**

也就是：

- `80%` 自动化
- `20%` 规则与人工控噪

完整流程可以压成：

- 原始来源
- 候选卡片
- 标准化卡片
- 规则校验
- 去重与归一
- 人工抽检
- 正式入库

### 7. 在当前几乎没有现成数据时，第一版 verifier-RAG 应从小规模启动

外部笔记和今天主线合并后的最重要结论是：

- 先做一个“小而能跑”的 RAG
- 不要等完整数据体系全齐了再开始

第一版建议优先用：

- `official_doc`
- `kernel_pattern_seed`
- 少量 `benchmark_note`

当前先不急着做：

- 大规模 `project_history`
- 服务化检索系统
- 复杂 reranker

### 8. 第一版 RAG 的推荐工程形态

在“从 0 启动”的条件下，今天补充确认了一套非常实用的轻量方案：

- 卡片存储：`jsonl`
- sparse 检索：`BM25`
- dense 检索：`FAISS`
- 融合与过滤：Python 脚本

并建议先建一套独立目录结构，例如：

- `raw_sources/`
- `workdir/`
- `indexes/`
- `retrieval/`
- `scripts/`

### 9. 第一版卡片不应依赖大模型生成

这一点今天很重要：

- 第一版 evidence card 最稳的做法不是直接让大模型大规模蒸馏
- 而是先用：
  - 模板
  - metadata
  - 规则

稳定生成第一批结构统一的卡片。

也就是说：

- `official_doc` 用 topic 模板
- `kernel_pattern_seed` 用标签模板
- `benchmark_note` 用规则模板

先把 schema 和检索路径打通，再考虑更复杂的蒸馏。

### 10. 训练与上线时的 RAG 使用方式

今天补充形成了比较完整的链路理解：

训练构造阶段：

- 原始样本
- 特征抽取
- 检索 top 3 cards
- teacher 参考原始样本 + top 3 cards 生成 verdict
- 最终训练样本只保留 top 1-2 条 `rag_retrieval`

上线推理阶段：

- candidate sample
- feature extractor
- retrieval query
- hybrid retrieval
- top 1-2 cards
- verifier
- verdict json

这和今天前面已经确认的 schema 完全兼容。

### 11. 合并后的实际执行建议

把外部笔记和今天项目内实现合并后，下一步更清晰了：

1. 先保持当前 intermediate JSONL 流程不变。  
2. 单独新建一个 verifier-RAG 的最小工程目录。  
3. 先人工 / 规则构造第一批 `kernel_pattern_seed` 和 `benchmark_note` 卡片。  
4. 用 probe queries 先调检索质量，再接 teacher。  
5. 等检索 top1/top3 基本靠谱后，再把 `rag_retrieval` 正式接回训练样本。  

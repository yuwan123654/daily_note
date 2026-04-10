# Daily Note

日期：2026-04-10

## 今日完成事项

- 分析了 `summary_table_level1.md` 的测试结果，并与历史结果做了对比。
- 结合生成的 kernel，分析了若干大误差样本的根因：
  - `14_Matmul_for_upper_triangular_matrices`
  - `51_Argmax_over_a_dimension`
  - `89_cumsum`
  - `95_CrossEntropyLoss`
- 说明了这些大误差主要来自生成 kernel 的语义实现错误，而不是简单浮点误差或 KernelBench 示例本身的问题。
- 给出了 `coder v1` 的改进方向，重点聚焦：
  - operator contract
  - reduction / scan / argmax / loss 等高风险语义错误族
  - `Verifier-v1` 和 `Coder-v2` 的后续价值
- 设计了 `Coder-v1` 的训练样本 schema，并落地为文档：
  - [coder_v1_training_schema.md](/Users/xuwanqi/Desktop/cudaRL/coder_v1_training_schema.md)
- 设计了“短 thinking / 结构化 contract”补训格式，并落地为文档：
  - [coder_v1_thinking_contract_format.md](/Users/xuwanqi/Desktop/cudaRL/coder_v1_thinking_contract_format.md)
- 参考现有 `example_pretty.md`，手工改写了 3 条样本作为新格式示例：
  - [example_pretty_thinking_contract.md](/Users/xuwanqi/Desktop/cudaRL/example_pretty_thinking_contract.md)
- 讨论了当前 `Coder-v1` 未开启 thinking 的影响，并判断：
  - 对当前版本有影响，但不是主瓶颈
  - 对后续 `Coder-v2 / patch` 阶段影响更大
- 评估了“在当前 `Coder-v1` 基础上补训并开启 thinking 冲终版 `Coder-v1`”的可行性，结论是可行，但应采用：
  - assistant 端保持 code-only
  - thinking 放在 user 侧结构化输入
  - 小剂量、分阶段、强验证的补训方案
- 给出了基于“当前已经在 `Qwen3-8B-Base` 上训了 1.5 epoch”的安全补训计划：
  - Stage A：`8000 / 0.5 epoch`
  - Stage B：`5000 / 0.6 epoch`
  - Stage C：`2500 / 0.3 epoch`
- 联合分析了：
  - [summary_table_level1.md](/Users/xuwanqi/Desktop/cudaRL/test/results/coder_v1/summary_table_level1.md)
  - [summary_table_level2.md](/Users/xuwanqi/Desktop/cudaRL/test/results/coder_v1/summary_table_level2.md)

## 关键结论

### 1. 当前 `Coder-v1` 的真实状态

- `level1`：
  - 成功率 `72.0%`
  - 平均 speedup `1.2726`
  - 基础算子能力已经成型
- `level2`：
  - 成功率 `59.0%`
  - 平均 speedup `0.8831`
  - 复杂算子组合、shape 变化、reduction 纪律仍明显不足

结论：

- 当前模型更适合被定义为 `Coder-v1-alpha`
- 尚不建议直接视为成熟的 `Coder-v1-final`

### 2. 当前最主要的失败模式

- `level1` 失败集中在：
  - reduction / norm / loss
  - scan / prefix
  - structured matmul
- `level2` 失败集中在：
  - shape mismatch
  - `Dimension out of range`
  - GroupNorm / LayerNorm / BatchNorm 与其他算子串联
  - ConvTranspose3d / ConvTranspose2d 复杂组合

### 3. “大误差”不是数值精度问题

大 `max_abs_err` 的主要来源是语义错误，例如：

- 求和区间写错
- flatten offset 映射错
- keepdim / output shape 规则错
- inclusive / exclusive scan 逻辑错
- loss reduction 语义错

### 4. thinking 的策略

- 不建议在 assistant 端输出 reasoning
- 推荐把 thinking 以短 checklist 的形式放入 user 侧输入
- 这样可以增强语义约束，又尽量不破坏 code-only 输出风格

## 已产出的关键文件

- [coder_v1_training_schema.md](/Users/xuwanqi/Desktop/cudaRL/coder_v1_training_schema.md)
- [coder_v1_thinking_contract_format.md](/Users/xuwanqi/Desktop/cudaRL/coder_v1_thinking_contract_format.md)
- [example_pretty_thinking_contract.md](/Users/xuwanqi/Desktop/cudaRL/example_pretty_thinking_contract.md)

## 后续建议

1. 优先补 `shape / keepdim / reduction` 纪律相关数据。
2. 针对 `argmax / cumsum / cross_entropy / structured matmul` 增加 contract + short-thinking 样本。
3. 用 `level1 + level2` 的 hard subset 做固定验证，不要只看总成功率。
4. 将当前模型继续定位为 `Coder-v1-alpha`，在补训完成并通过 hard set 后再考虑升级为正式 `Coder-v1`。

## 补充：旧版 `results_ex` 与当前版 `coder_v1` 的对比结论

今天额外对比了：

- 旧版：长思维链 + few-shot + thinking 模式训练后的 `level1_ex`
- 当前版：当前 `coder_v1` 的 `level1`

对比文件：

- [summary_table_level1_ex.md](/Users/xuwanqi/Desktop/cudaRL/test/results_ex/summary_table_level1_ex.md)
- [summary_table_level1.md](/Users/xuwanqi/Desktop/cudaRL/test/results/coder_v1/summary_table_level1.md)

并整理了正式分析文档：

- [results_ex_vs_current_level1_analysis.md](/Users/xuwanqi/Desktop/cudaRL/results_ex_vs_current_level1_analysis.md)

### 关键量化结果

在两版共同覆盖的 `79` 个 level1 题目上：

- 旧版：
  - compile 通过：`53`
  - success：`27`
- 当前版：
  - compile 通过：`72`
  - success：`53`

变化：

- compile：`53 -> 72`
- success：`27 -> 53`
- 修好的题：`27`
- 回归题：`1`

### 对提升原因的判断

当前版明显提升的主要原因，不是“模型更会思考”，而是：

1. 输出分布更贴近 benchmark 的真实目标  
   当前版更专注生成最终代码，而不是学习长思维链和 few-shot 的表达风格。

2. 接口与代码骨架更稳定  
   `ModelNew`、`load_inline`、参数签名、模块结构保留都更稳，因此 compile 层提升显著。

3. 更少被 few-shot 示例误导  
   旧版更容易把示例中的局部实现策略迁移到不匹配的题目上，导致接口不兼容、维度越界、参数不匹配等问题。

4. 更少写出“形式正确但实际低效”的实现  
   当前版在不少共同成功题上速度明显好于旧版，说明它更少依赖低效模板实现。

### 对训练策略的启示

- 对 `Coder-v1` 这类 benchmark 型代码生成任务，不建议再把“长思维链 + heavy few-shot”作为主训练路线。
- 更推荐：
  - assistant 端继续保持 code-only
  - user 端注入 `operator contract`
  - 只在 hard case 上加入短 thinking checklist

这次对比进一步支持了今天前面形成的判断：

- 当前阶段最重要的是“对齐最终代码输出目标”
- 而不是继续增加长思维链或示例风格强度

## 补充：GPT摘要

今天还额外汇总并吸收了另一份笔记：

- 来源文件：[daily_note.md](/Users/xuwanqi/Downloads/daily_note.md)

这份笔记的内容和今天主线高度一致，但补充了几块更偏“训练路线设计”和“工程落地”的结论，值得并入当前项目记录。

### 1. 关于 Verifier：优先 DPO

外部笔记对 Verifier 训练路线的建议很明确：

- `Verifier SFT` 后更推荐 `DPO`
- 不优先 `ORPO`

原因：

- 目标是让 Verifier 更稳地输出：
  - `failure_type`
  - `diagnosis`
  - `evidence`
  - `decision`
  - `patch_spec`
- DPO 更适合这种“在已有 schema 基础上做偏好精修”的任务

推荐的第一轮 DPO 数据量：

- `1200–1800` 对高质量 pair

### 2. 关于 DPO / GRPO 的成本认知

外部笔记也补充了一个重要工程判断：

- `DPO` 是离线偏好优化
- `GRPO` 是在线 RL
- 在代码 / CUDA 场景里，GRPO 需要：
  - 在线生成
  - 编译
  - 执行
  - 测试
  - profiling

所以：

- **DPO 的时间与工程成本通常明显低于 GRPO**
- 这也进一步支持了当前阶段：
  - Verifier 先 SFT + DPO
  - Coder 先补训
  - RL 放后面

### 3. 与今天主线合并后的最终判断

把这份外部笔记和今天项目内分析合并后，可以更加明确地确认：

- 当前主线没有偏：
  - **Verifier-first**
  - **Coder 拆成 v1 / v2**
  - **assistant 端坚持 code-only**
  - **先补训，再 RL**
- 训练工作的优先级已经比较清楚：
  1. 失败样本分桶
  2. Stage A / B 的 contract 数据
  3. Verifier DPO pair
  4. 固化 Verifier schema

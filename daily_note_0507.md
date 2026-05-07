# Daily Note — 2026-05-07

## 今日形成的关键决策

今天主要围绕 Phase 1 Verifier 的冷启动数据、taxonomy、history schema 和后续训练路线做决策。下面只记录已经确定或强建议采用的结论。

---

## 1. Phase 1 的定位

Phase 1 不负责证明最终策略切换能力，只负责冷启动一个可接入闭环的 Verifier。

Phase 1 过关标准应聚焦：

- 输出 schema 稳定
- diagnosis 基本可信
- patch_spec 能给 Coder 提供可执行修改方向
- decision 不塌缩

`decision_accuracy` 不应作为 Phase 1 的核心硬指标；真正的策略选择能力放到 Phase 3/5 通过闭环轨迹和 DPO 验证。

---

## 2. Backbone 选择

主线建议使用：

```text
Qwen3-8B-Instruct
```

Base 可以作为 ablation，而不建议作为当前主线。

理由：

- 当前 3600 条 Phase 1 数据不足以让 Base 稳定学会复杂结构化输出
- Phase 1 是冷启动，不是论文核心贡献
- Instruct 能更快获得稳定 schema，便于进入 Phase 3/5
- 只要所有 verifier-side baseline 使用同一 backbone，论文公平性仍然成立

---

## 3. Thinking 策略

Phase 1 主线不开启 Qwen thinking mode。

采用：

```text
enable_thinking = false
保留显式 ## Reasoning
```

原因：

- 当前最重要的是格式稳定
- thinking mode 可能带来输出变长、截断、parser 污染
- 论文需要的是显式 policy-level reasoning 字段，而不是依赖 hidden thinking

---

## 4. Taxonomy 调整

当前数据中的 `stagnation` 实际表示“正确且性能已经很好，可以停止”，这和论文中需要的 true stagnation 不一致。

决定将 taxonomy 调整为：

```text
diagnosis ∈ {
  compile_error,
  correctness_error,
  performance_bottleneck,
  stagnation,
  accept
}
```

语义：

```text
accept:
当前实现正确，性能达到阈值，继续优化收益不值得。
decision = stop

stagnation:
多轮优化无明显收益，当前路线卡住。
decision = rewrite / stop
```

后续应将当前：

```text
stagnation + stop
```

改为：

```text
accept + stop
```

真正的 `stagnation + rewrite` 需要通过 proxy history 或 Phase 3 真实闭环轨迹补充。

---

## 5. Patch vs Rewrite 边界

今天确定的 decision 语义：

```text
patch = 当前实现主体路线是对的，只需要局部修
rewrite = 当前实现核心路线错了，继续局部修补收益低
```

选择 `patch`：

- compile/binding/init/dtype/shape 等局部错误
- wrapper / launch overhead
- small workload overhead
- redundant memory ops
- bad_library_path
- speedup 接近 baseline，仍有局部修复空间

选择 `rewrite`：

- semantic mismatch
- 核心算子语义错
- reduction 维度/计算顺序根本错
- 严重 bad_parallelization
- 严重 memory_bound_access
- 需要换并行策略、tiling、memory layout、host path

---

## 6. Performance Bottleneck vs Stagnation

区分原则：

```text
performance_bottleneck = 当前单轮实现有性能瓶颈
stagnation = 多轮优化过程没有进展
```

没有 trajectory history 时，不应把单轮慢样本硬标为 `stagnation`。

单轮正确但慢：

```text
diagnosis = performance_bottleneck
decision = patch / rewrite
```

多轮 patch 无收益：

```text
diagnosis = stagnation
decision = rewrite
```

已经足够好：

```text
diagnosis = accept
decision = stop
```

---

## 7. Trajectory History / Signals 设计

后续所有 Verifier 输入都应包含：

```text
## Trajectory History
## Trajectory Signals
```

即使是第一轮，也使用固定空历史：

```text
## Trajectory History
- No prior turns available. This is a single-state cold-start sample.

## Trajectory Signals
- history_type: none
- turns_available: 0
- current_turn: 1
- best_speedup_so_far: None
- turns_without_best_improvement: 0
```

原因：

- 保持 Phase 1 / Phase 3 schema 连续
- 避免模型把“是否出现 history”当成 diagnosis 捷径
- 后续真实闭环采样中，每一轮都能自然填入真实 history

---

## 8. 冷启动数据中的 History 策略

不应给每条冷启动样本都构造假 history。

推荐比例：

```text
70-80%:
No prior turns / first_turn

20-30%:
proxy history / proxy signals

0-5%:
proxy_stagnation + rewrite
```

原则：

- 所有样本都有 History / Signals 字段
- 大多数 single-state 样本保持 first-turn
- 少量 proxy history 用来适应后续闭环输入格式
- true stagnation 必须有 history，不应是 empty-history
- Phase 3 后用真实 trajectory 数据替代 proxy-history 的权重

---

## 9. 已完成的数据处理决策

今天已经对 `phase1_verifier_sft_final.jsonl` 做了两类处理：

1. 清除 `optimized_code` / `optimized target` 泄漏，并统一 assistant 输出模板。
2. 只更新 `## Verifier Verdict` 的核心标签，不联动修改 `## Reasoning`。

当前已保留备份：

```text
phase1_verifier_sft_final.jsonl.bak_pre_clean_opt_refs
phase1_verifier_sft_final.jsonl.bak_pre_relabel_verdict
```

也已从 merged parquet 构造补充集：

```text
phase1_supplement.jsonl
```

该补充集只作为待筛选、待 teacher-fill 的候选池，不建议未筛选全量混入训练。

---

## 10. 后续执行计划

短期优先事项：

1. 修改 schema 和 eval 脚本：
   - 添加 `accept`
   - 更新 diagnosis 约束
   - 调整 Phase 1 gate

2. 更新数据：
   - 将当前 `stagnation + stop` 改为 `accept + stop`
   - 为主数据和补充数据统一加入 `Trajectory History / Trajectory Signals`

3. 构造少量 proxy-history 数据：
   - 重点补 `stagnation + rewrite`
   - 不要大规模伪造 history

4. 对 `phase1_supplement.jsonl` 做筛选和平衡：
   - 优先选 `performance_bottleneck + patch`
   - 补 `correctness_error + patch`
   - 补 compile subtype 多样性
   - 保留 accept/stop 边界样本

5. 再进行 Instruct SFT：
   - `enable_thinking=false`
   - `lr=1e-6 ~ 2e-6`
   - `epoch=1-2`
   - held-out task split 评估


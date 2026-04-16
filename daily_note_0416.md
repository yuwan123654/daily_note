# Daily Note 0416

## 今日目标

围绕 joint master 数据构建流程，先完成 merged parquet 的标准化，再控制 teacher 包装成本，筛出更适合 `Verifier v1 + Coder v2` 的高价值子集，并为后续多路并发包装准备脚本与运行方案。

## 主要工作

### 1. joint master 第一步完成：merged parquet 标准化

- 新增脚本：[build_joint_base_rows.py](/Users/xuwanqi/Desktop/cudaRL/datasets/build_joint_base_rows.py)
- 新增流程文档：[joint_master_build_plan.md](/Users/xuwanqi/Desktop/cudaRL/datasets/joint_master_build_plan.md)
- 将 merged parquet 转为标准化单条记录，产出：
  - [joint_base_rows.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_base/joint_base_rows.jsonl)
  - [joint_base_rows.parquet](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_base/joint_base_rows.parquet)
  - [joint_base_rows_summary.json](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_base/joint_base_rows_summary.json)

标准化结果概览：

- 总行数：`30615`
- `benchmark_ready`: `16457`
- `correctness_failed`: `6715`
- `compile_failed`: `7441`
- `runtime_missing`: `2`

这一步统一了后续需要的核心字段，包括：

- `sample_id`
- `task_id / task_name / kernel_name / level_id`
- `reference_task_code`
- `candidate_cuda_code`
- `metrics`
- `raw_artifacts`
- `row_status`
- `error_bucket`

### 2. candidate CUDA 包装脚本搭建完成

- 新增 prompt：[joint_base_to_python_wrapper_prompt.txt](/Users/xuwanqi/Desktop/cudaRL/datasets/joint_base_to_python_wrapper_prompt.txt)
- 新增脚本：[build_joint_wrapped_candidates.py](/Users/xuwanqi/Desktop/cudaRL/datasets/build_joint_wrapped_candidates.py)

包装约束：

- 保留 `class Model`
- 新增 `class ModelNew`
- `ModelNew.__init__` 与 `Model.__init__` 完全一致
- `ModelNew.forward` 与 `Model.forward` 完全一致
- 不允许出现 `get_inputs()` / `get_init_inputs()`
- 必须使用 `load_inline`

脚本输出定义已经明确：

- 成功样本：`joint_wrapped_candidates.jsonl`
- reject 样本：`rejected_joint_wrapped_candidates.jsonl`
- 全量日志：`generation_log.jsonl`
- 可选 parquet：`joint_wrapped_candidates.parquet`

### 3. pilot 自动质检脚本补齐

- 新增脚本：[qc_joint_wrapped_candidates.py](/Users/xuwanqi/Desktop/cudaRL/datasets/qc_joint_wrapped_candidates.py)

支持在 pilot 跑完后自动检查：

- 成功样本二次结构校验
- 高频 reject reason 汇总
- API / parse error 汇总
- success / reject / log 三份文件的一致性
- `with_cuda=True`、`verbose=False` 等代码 warning
- 输出 `json + md` 两份质检报告

### 4. 训练成本控制：先筛选再包装

确认全量包装 `30615` 条成本过高，因此先从 [joint_base_rows.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_base/joint_base_rows.jsonl) 中筛出更适合 `Verifier v1 + Coder v2` 的高价值子集。

- 新增筛选脚本：[filter_joint_packaging_pool.py](/Users/xuwanqi/Desktop/cudaRL/datasets/filter_joint_packaging_pool.py)

先做了一版偏优化训练的默认筛选，再进一步调成更偏 `Coder v2` 的配置：

- `compile_failed`: 从 `600` 调低到 `450`
- `performance_bottleneck`: 从 `1800` 提高到 `2200`
- `target_pool`: 从 `1600` 提高到 `1889`
- `correctness_failed`: 保持 `1000`
- `stagnation`: `300`

最终当前主用筛选结果：

- 输出目录：[output_joint_filter_aggressive](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive)
- 主文件：[joint_packaging_pool.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive/joint_packaging_pool.jsonl)
- 总量：`5537`
- 任务数：`89`
- `compile_failed`: `450`
- `correctness_failed`: `1000`
- `performance_bottleneck`: `2200`
- `stagnation`: `300`
- `target_pool`: `1889`

关键结论：

- 当前筛选结果已经非常偏向“有优化闭环的任务”
- aggressive 版相比宽松版几乎不怎么缩量，但语义上更干净，更适合直接服务 `Coder v2`
- 说明当前预算主要已经集中在“慢样本 + 强 target”的任务上

### 5. 支持 optimization closure 的 aggressive 筛选

在筛选脚本中加入了：

- `--require-optimization-closure`
- `--min-perf-per-task-for-closure`
- `--min-target-per-task-for-closure`

语义是只保留：

- 同任务内存在 `performance_bottleneck`
- 同任务内也存在 `target_pool`

这样能保证后续 teacher 包装和 pair 构造更贴近 `Coder v2` 的真实闭环。

### 6. aggressive 子集切成 5 份并发处理

将 `5537` 条 aggressive 子集切成 5 份，用于并发请求 teacher：

- 输出目录：[shards_5](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive/shards_5)
- 清单文件：[manifest.json](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive/shards_5/manifest.json)

5 个 shard：

- [joint_packaging_pool.part_01_of_05.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive/shards_5/joint_packaging_pool.part_01_of_05.jsonl)
- [joint_packaging_pool.part_02_of_05.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive/shards_5/joint_packaging_pool.part_02_of_05.jsonl)
- [joint_packaging_pool.part_03_of_05.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive/shards_5/joint_packaging_pool.part_03_of_05.jsonl)
- [joint_packaging_pool.part_04_of_05.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive/shards_5/joint_packaging_pool.part_04_of_05.jsonl)
- [joint_packaging_pool.part_05_of_05.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive/shards_5/joint_packaging_pool.part_05_of_05.jsonl)

分片策略：

- `round_robin_preserve_global_order`

每片大小：

- `1108 / 1108 / 1107 / 1107 / 1107`

这样比顺序截断更均匀，适合多路并发 teacher 包装。

### 7. 5 路并发 teacher 运行脚本

- 新增脚本：[run_joint_wrapped_aggressive_5way.sh](/Users/xuwanqi/Desktop/cudaRL/datasets/run_joint_wrapped_aggressive_5way.sh)

脚本能力：

- 默认读取 5 个 shard
- 并发启动 5 个 `build_joint_wrapped_candidates.py`
- 每一路写到独立输出目录
- 每一路写独立 `run.log`
- 支持 `RESUME=1`
- 支持 `PILOT=1`
- 支持 `WRITE_PARQUET=1`

输出目录默认：

- `/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_wrapped_aggressive/part_01`
- ...
- `/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_wrapped_aggressive/part_05`

### 8. 支持 5 个分片分别使用不同 API key

并发脚本进一步支持多 key 分摊：

- `API_KEY_01`
- `API_KEY_02`
- `API_KEY_03`
- `API_KEY_04`
- `API_KEY_05`

回退逻辑：

- 如果某一路未设置专属 key，则回退到通用 `API_KEY`

这样可以：

- 降低单 key 限流压力
- 提高 5 路并发稳定性
- 在多 provider / OpenAI-compatible 平台下更容易跑满吞吐


## 今日关键判断

- 全量包装 merged 数据成本过高，必须先筛选再包装
- 当前 aggressive 筛选集已经足够偏向 `Coder v2` 的优化闭环任务
- `5537` 条切成 5 片并发跑 teacher 是更现实的路径

## 当前产物总览

- joint base 标准化：
  - [build_joint_base_rows.py](/Users/xuwanqi/Desktop/cudaRL/datasets/build_joint_base_rows.py)
  - [joint_base_rows.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_base/joint_base_rows.jsonl)
- 包装脚本与 prompt：
  - [build_joint_wrapped_candidates.py](/Users/xuwanqi/Desktop/cudaRL/datasets/build_joint_wrapped_candidates.py)
  - [joint_base_to_python_wrapper_prompt.txt](/Users/xuwanqi/Desktop/cudaRL/datasets/joint_base_to_python_wrapper_prompt.txt)
- pilot 质检：
  - [qc_joint_wrapped_candidates.py](/Users/xuwanqi/Desktop/cudaRL/datasets/qc_joint_wrapped_candidates.py)
- aggressive 筛选：
  - [filter_joint_packaging_pool.py](/Users/xuwanqi/Desktop/cudaRL/datasets/filter_joint_packaging_pool.py)
  - [joint_packaging_pool.jsonl](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive/joint_packaging_pool.jsonl)
- 5 分片并发：
  - [manifest.json](/Users/xuwanqi/Desktop/cudaRL/datasets/output_joint_filter_aggressive/shards_5/manifest.json)
  - [run_joint_wrapped_aggressive_5way.sh](/Users/xuwanqi/Desktop/cudaRL/datasets/run_joint_wrapped_aggressive_5way.sh)

## 下一步

- 等 5 路 teacher 包装完成
- 合并 5 路 `joint_wrapped_candidates`
- 跑统一 QC
- 再进入：
  - `task_contract` 抽取
  - `execution_evidence` 规范化
  - `verdict / coder_target` 联合样本构造

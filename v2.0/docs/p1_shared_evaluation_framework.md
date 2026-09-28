# P1 共享评估框架（Issue #6）

本框架把数据访问、分组划分、指标聚合和输出格式固定为一个模型无关接口。SSS、fCO2、TA 及后续模型不得自行读取完整标签表后再切分，也不得自行定义同名指标。

## 数据访问边界

`P1DataGateway` 使用 parquet predicate pushdown，只把当前用途允许的 split 读入进程：训练只能读取 `train`，模型选择只能读取 `development`。`locked_test` 需要显式创建并落盘的 `LockedTestGrant`，`external_independent` 在 P1 阶段没有开放入口。预测接口不返回标签。SSS/fCO2 标签止于 2025，TA/DIC 标签止于 2024；2026 只能作为 `prediction_status=provisional` 的预测行。

正式运行前执行完整哈希验证：

```powershell
.\.venv\Scripts\python.exe scripts\p1_framework.py audit --hash-mode full --output outputs\audits\p1_framework_v2.2.json
```

开发 smoke test 可以用 `--hash-mode artifacts`，但生成的审计会明确记录未重新哈希原始大文件，不能冒充正式运行。

## 统一协议

- 五折 CV 只在 `train` 内按 `group_key` 整航次留出。
- 空间转移通过 `leave_group_out(..., "lme_id")`，环境制度转移使用 `regime_id`。
- 前向链固定为航次最大年份 `<=2018`、`2019–2021`、`2022–2025`，通过 `load_labels(..., split_scheme="forward")` 读取；它同样在 parquet 层只物化当前阶段的航次组。
- batch 采用 region → cruise → month → record 的确定性均衡抽样。
- checkpoint 默认最小化 development LME-macro RMSE；最终汇总必须包含种子 100、101、102。
- 指标统一报告 pooled、cruise-equal、LME macro、regime macro、worst LME 和 support-distance 分层。`r2` 是 `1-SSE/SST`，Pearson 相关另列。

预测表由 `standardize_predictions` 校验，包含目标/模型谱系、不确定性及区间、支持距离、OOD、split/fold、区域、种子和 provisional 状态。训练脚本应把每次 run 的 audit JSON、标准预测 parquet、指标 parquet 和 checkpoint selection JSON 一起保存到被 `.gitignore` 排除的 `outputs/`。

配置入口为 `configs/p1_framework_v2.2.yaml`。查看某个可访问数据层而不加载其他 split：

```powershell
.\.venv\Scripts\python.exe scripts\p1_framework.py describe fco2 train
.\.venv\Scripts\python.exe scripts\p1_framework.py describe ta selection
```

# P1.1 沿海 SSS 产品可行性实验

> Reviewer-ready 的完整方法、图片、图片源表和哈希清单已归档到 [`docs/experiment_archive/p1_sss_viability_v2.2/`](experiment_archive/p1_sss_viability_v2.2/REPORT.md)。本文保留为项目内简明结论入口。

日期：2026-09-28。实验代码提交：`dc653dd`。数据清单：`data_manifest_v2.2.json`，SHA256 为 `46b01085fbb577b24dbb62b6ff7b7a53a5d402d6ab93d4c076d67f37f32e9e9f`。

本实验只读取冻结的 train 和 development 标签。`locked_test` 与 `external_independent` 均未打开。结论的地理范围限于冻结缓存覆盖的北美邻近沿海水域（0–70.125°N、180–315°E），不能据此给出全球沿海 SSS 的通过结论。

## 数据与运行

- Train：330,961 个有效 GLORYS 背景匹配记录，2,255 个航次组。
- Development：71,624 个记录，479 个航次组，覆盖至 2025。
- 六类候选：GLORYS、区域月份偏差、Ridge residual、CatBoost residual、point MLP residual、top-2 soft experts residual。
- 神经模型每次 5,000 optimizer steps，batch size 2,048；CatBoost 上限 1,500 trees。
- 随机候选使用种子 100、101、102；另运行五折 cruise CV 和独立的 forward-chain development。
- 完整正式运行耗时约 504 秒，所有 checkpoint、预测、指标和哈希保存在本地 `outputs/experiments/p1_sss_viability_v2.2/`，不进入 Git。

## Development 结果

| 模型 | Pooled RMSE | Cruise-equal RMSE | LME-macro RMSE | Worst-LME RMSE | Pooled skill vs GLORYS |
|---|---:|---:|---:|---:|---:|
| GLORYS | 1.776 | 2.607 | 2.629 | 9.184 | 0.000 |
| Regional-month bias | 1.342 | 2.301 | 2.124 | 8.766 | 0.428 |
| Ridge residual | 1.381 | 1.983 | 2.200 | 9.710 | 0.395 |
| CatBoost residual | **0.973** | **1.392** | 1.781 | 9.613 | **0.700** |
| Point MLP residual | 1.039 | 1.585 | 1.591 | **6.367** | 0.658 |
| Soft experts residual | 1.035 | 1.567 | **1.574** | 6.822 | 0.660 |

冻结的 checkpoint 选择指标是 development LME-macro RMSE，因此主候选为 `soft_experts_residual`。它相对 GLORYS 将 cruise-equal RMSE 降低 39.9%，LME-macro RMSE 降低 40.1%，worst-LME RMSE 降低 25.7%。三个种子的 pooled skill 均为正。

CatBoost 在 pooled、cruise-equal 和五折 CV 上更强，但 development LME-macro 较差。差异主要来自仅 30 条 development 记录的 LME 55：CatBoost 在该 LME 的 skill 为 -0.097，soft experts 为 +0.444。因为选择规则已在运行前冻结，不能在看见结果后改选 CatBoost；它保留为 #11 的预声明敏感性对照，不能根据 locked 结果再反向选择模型。

## 五折 cruise CV 与 forward chain

| 模型 | CV LME-macro RMSE | CV pooled skill |
|---|---:|---:|
| GLORYS | 2.027 ± 0.131 | 0.000 |
| CatBoost residual | **1.222 ± 0.150** | **0.681 ± 0.045** |
| Point MLP residual | 1.286 ± 0.107 | 0.604 ± 0.032 |
| Soft experts residual | 1.293 ± 0.122 | 0.603 ± 0.033 |

Forward chain 使用航次最大年份 ≤2018 训练、2019–2021 development。Soft experts 三种子的 pooled skill 为 0.596 ± 0.063；LME-macro RMSE 为 1.815 ± 0.070 PSU，而 GLORYS 为 2.670 PSU。三个种子均保持正 forward skill。

## 区域、低盐和支持距离

- 在 development 中，14/14 个样本数不少于 100 的 LME 都获得正 skill。
- SSS <20 的 57 个记录：RMSE 从 19.405 降至 13.264 PSU，skill 0.533。
- SSS 20–30 的 2,156 个记录：RMSE 从 5.823 降至 3.633 PSU，skill 0.610。
- 所有预定义盐度区间均为正 skill。
- 另外计算了真正的 SSS 训练位置支持距离，没有误用共享缓存中的 `nearest_carbon_km`。从 0–1 km 到 >250 km 的所有距离带均为正 skill；>250 km 只有 55 条记录，skill 为 0.435，应视为不确定的远外推诊断。

Development 校准得到 90% 绝对残差半宽 1.156 PSU，在同一 development 集覆盖率为 0.900。该数值只是待检验的校准参数，不是独立覆盖证据；真实覆盖率必须由 #11 locked test 检验。

## 决定

Issue #7 的全部 development 门槛通过，`soft_experts_residual` 被提名进入 Issue #11 的一次性 locked gate。当前状态是“regional candidate nominated”，还不是 `pass_regional`。最终状态只能在 #11 中从 `pass_regional`、`diagnostic_only` 或 `fail` 中确定。全球 SSS 结论仍需要全球冻结缓存和后续外部 SOCAT 增量验证。

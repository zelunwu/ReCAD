# P0 v2.2 数据覆盖更正与冻结报告

v2.2 修正了 v2.1 把“观测标签截止”“训练核心期”和“产品时间轴”混为一谈的问题。数据资产保留 1993–2026；SSS/fCO2 的核心观测与训练期为 1993–2025；TA/DIC 观测标签到 2024；2025–2026 的 TA 可以由模型前向预测，DIC 可以由预测的 T、SSS、fCO2、TA 经 inverse CO2SYS 推导。2026 是滚动不完整年份，所有结果必须带 `provisional` 标记。

机器可读入口为 `configs/frozen/data_manifest_v2.2.json`，SHA256 为 `46b01085fbb577b24dbb62b6ff7b7a53a5d402d6ab93d4c076d67f37f32e9e9f`。v2.1 manifest 和预登记记录保留作审计，但已被 v2.2 替代，禁止启动新的正式实验。

## 2025 与 2026 覆盖

2025 的 SST、SSS、ADT、风速、大气 xCO2/pCO2 和 SOCAT fCO2 均覆盖 12 个月。SOCAT fCO2 有 40,110 个沿岸格月，北美 compact cache 新增后共有 522,322 个 `cruise × grid × month` 记录。因此 2025 进入正常的 cruise-grouped train/development/locked-test 分配，不作为独立测试。

2026 已保留在产品轴上，但各输入更新进度不同：SST 8 个月、SSS 6 个月、ADT 1 个月、风速 7 个月、大气 CO2 5 个月、SOCAT fCO2 1 个月且仅 36 个全球沿岸格月。严格要求全部输入时，目前只有 1 月可推理，共 140,746 个沿岸节点；2–12 月严格可推理节点为 0。缺失感知模型将来可以扩大覆盖，但必须保留逐变量 availability 和 provisional 状态。

`C:\backup\phd\data\processed\recad_v2_2\inference_availability_v2.2.nc` 逐年、逐月、逐沿岸节点保存每个 predictor 的可用性和 `strict_all_inputs_ready`。它描述前向推理能力，不代表存在 TA/DIC 或 fCO2 验证标签。

## 标签与预测 provenance

| 参数 | 观测监督核心期 | 产品/推理时间轴 | 2025–2026 provenance |
|---|---|---|---|
| SSS | SOCAT 1993–2025 | 至 2026 可用输入月份 | `model_predicted`，有 SOCAT 时可评价 |
| fCO2 | SOCAT 1993–2025 | 至 2026 可用输入月份 | `model_predicted`，有 SOCAT 时可评价 |
| TA | CODAP/GLODAP 至 2024 | 至 2026 可用输入月份 | `model_predicted`，没有观测时不得声称验证通过 |
| DIC | CODAP/GLODAP 至 2024 | 至 2026 可用输入月份 | `co2sys_derived_from_predictions` |

碳表中 6,096 条由其他碳参数计算的 fCO2 继续保持 audit-only，不能作为独立监督。实测/QC 合格的碳表 fCO2 只有 293 条；fCO2 的主要监督仍来自 SOCAT。

## Split 与资产

v2.2 有 3,910 个唯一航次组：train 2,713、development 571、locked test 585、external independent 41。主 split、五折 grouped CV 和 forward split 的跨组违规均为 0。2025 的航次遵循同一哈希分组；任何跨 2025–2026 的航次会整体归入 `provisional_2026`，防止航次泄漏。

大文件位于 `C:\backup\phd\data\processed\recad_v2_2`：

- `na_socat_cache_v2.2.parquet`：522,322 个北美 SOCAT 航次格月；
- `na_carbon_cache_v2.2.parquet`：13,501 条 primary 碳观测；
- `inference_availability_v2.2.nc`：2025–2026 前向推理覆盖；
- `spatial_support_v2.2.nc`：LME、海盆、环境区和观测支持；
- `coastal_graph_v2.2.npz`：148,795 节点、492,570 条水体四邻接有向边；
- `split_manifest_v2.2.parquet` 与 `coverage_independence_audit_v2.2.json`。

复现与验证：

```powershell
.\.venv\Scripts\python.exe scripts\build_p0_freeze_assets.py
.\.venv\Scripts\python.exe scripts\verify_p0_freeze.py
.\.venv\Scripts\python.exe -m pytest tests\test_p0_freeze_assets.py tests\test_weighted_carbonate_selection.py -q
```

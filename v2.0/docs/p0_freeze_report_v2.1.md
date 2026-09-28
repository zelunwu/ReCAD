# P0 v2.1 数据冻结与验证资产报告

冻结时间为 2026-09-28，正式数据期为 1993-01 至 2024-12。2025 为统一完整窗口的保守排除年；2026 在冻结时不完整，禁止进入训练、选择和测试。机器可读入口是 `configs/frozen/data_manifest_v2.1.json`，其 SHA256 为 `5980f2e31a78f7dcc9bebbb6508c101de3298f30e722ca0b6a8c9116591bb725`。

## 冻结资产

大文件位于 `C:\backup\phd\data\processed\recad_v2_1`，不进入 Git：

| 资产 | 内容 |
|---|---|
| `spatial_support_v2.1.nc` | 148,795 个沿岸节点的 LME、粗海盆、12 个目标无关环境区、碳观测/航次数和最近支持距离 |
| `coastal_graph_v2.1.npz` | 148,795 节点、492,570 条有向四邻接水体边；允许经度环绕，禁止跨陆地连边 |
| `split_manifest_v2.1.parquet` | SOCAT 与碳观测的统一航次组、主 split、五折 grouped CV 和 forward split |
| `na_socat_cache_v2.1.parquet` | 507,812 条 `cruise × grid × month` 北美 fCO2/现场盐度记录及同月 predictor |
| `na_carbon_cache_v2.1.parquet` | 13,501 条北美 primary 碳观测、QC、方法来源、predictor 和空间标签 |
| `coverage_independence_audit_v2.1.json` | split、覆盖、泄漏和图结构验收结果 |

所有输入和上述产物均在 data manifest 中登记绝对路径与 SHA256。`prepared_global_p32.nc` 的冻结哈希为 `13ac840c3bbce0a0f8c51cf787cb54310a7065b0fd0b07aeaa9925d15ac47719`。

## Split 与独立验证

主 split 完全由标准化 Expocode/cruise key 的 salted SHA256 决定，不读取目标值。共有 3,811 个唯一组：train 2,623、development 556、locked test 591、external independent 41。五折 CV 的组数为 771/767/809/726/738。主 split、CV 和 forward split 的跨组泄漏均为 0。

碳参数外部集由 CODAP-NA 中 2022–2024 且未匹配到 GLODAPv2.2023 的 41 个完整航次组成。它们在 SOCAT 和碳缓存中统一标记为 `external_independent`，禁止用于训练、调参、checkpoint 选择和消融决策。SSS/fCO2 的外部集锁定为未来 SOCAT 版本相对 v2026 的新增航次和 2025 年后观测；冻结时标签不可用，P3 前不得下载或查看。

## 标签与信息泄漏

北美碳缓存中 good 标签数为 SSS 13,372、TA 12,686、DIC 12,324，TA+DIC 同观测 11,509。fCO2 方法审计发现 6,096 条为计算值，仅 293 条满足 QC 且方法为实测/二次质控。计算 fCO2 只保留用于来源审计，不作为独立 fCO2 监督；四参数均为独立实测且共址的记录只有 157 条。这一结果支持继续用 SOCAT 监督 fCO2，并把 CO2SYS 作为有支持度的结构约束或派生诊断，而不是把计算 fCO2 重复当真值。

环境区只使用经纬度与 SST、SSS 背景、ADT、风速和大气 pCO2 的 1993–2024 均值/变率，不使用 fCO2、TA、DIC 或观测密度。LME 覆盖 118,830 个沿岸节点；LME 外节点仍有粗海盆和环境区标签，可用于 OOD 与 worst-group 报告。

## 工程修复与复现

`scripts/run_weighted_carbonate.py` 原先调用未定义的 `selection_score`，完整训练会在第一次 checkpoint 选择时失败。现在选择分数等权组合 SSS/fCO2 与 TA/DIC 两个目标族的标准化 RMSE，任何一侧退化都会受到惩罚。回归测试同时覆盖非有限 checkpoint 拒绝和沿岸图不跨陆地。

重建命令：

```powershell
.\.venv\Scripts\python.exe scripts\build_p0_freeze_assets.py
.\.venv\Scripts\python.exe -m pytest tests\test_p0_freeze_assets.py tests\test_weighted_carbonate_selection.py -q
```

P1 的 SSS、fCO2 和 TA 首轮实验已分别预登记在 `configs/experiment_records/`。任何模型配置、训练预算和源码哈希必须在实际启动前补齐；修改数据、split、外部组或截止年必须提升 manifest 版本，不能覆盖 v2.1。

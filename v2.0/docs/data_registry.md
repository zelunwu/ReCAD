# ReCAD v2 数据总账

本文件记录进入实验的信息资产。当前正式入口是 `configs/frozen/data_manifest_v2.2.json`；数据轴保留到 2026，SSS/fCO2 核心期到 2025，TA/DIC 观测标签到 2024。v2.1 因错误排除完整的 2025 数据而被替代。

## 原始和标准化输入场

| 数据 | 当前版本/时间 | 本机位置或产物 | 用途 | 状态与限制 |
|---|---|---|---|---|
| SOCAT coastal | v2026；逐观测表约 2050 万行 | 原始目录见 `data_download.md`；标准化后进入 prepared/joint cache | fCO2、现场 SSS | 2026 当前仅 36 个格月，正式产品应截止 2025 或最后完整年；2004–2005 已打开 |
| SST | 1993–2026 月场 | `outputs/prepared_global_p32.nc` | 全部任务输入 | 2025 完整；2026 当前到 8 月；文件 SHA256 `13ac840c…47719` |
| GLORYS SSS | 1993–2026 月场 | 同上 | SSS 背景、TA mixing 输入 | SSS 产品是偏差订正；独立盐度可能已被再分析同化 |
| ADT/SSH | 1993–2026 月场 | 同上 | 环流/中尺度代理 | 数据版本和哈希待冻结 |
| CCMP 风速 | 1993–2026 月场 | 同上 | fCO2/通量过程输入 | 日文件变量 `ws`、月文件变量 `w`；处理代码已兼容 |
| 大气 CO2/pCO2air | NOAA GML 路线 | 同上 | fCO2 趋势和海气梯度 | MBL 年度产品会重算，正式 manifest 必须固定下载版本 |
| MODIS Chl-a | Aqua 2002-07–2020-12 | `outputs/cache_naccom/chla.nc` | 生物过程输入 | NACCOM 已标准化；全球 Chl-a 尚未构建；模型必须显式处理缺失 |
| LME/环境区/观测支持 | LME 66；12 个 predictor-only 环境区 | `C:\backup\phd\data\processed\recad_v2_2\spatial_support_v2.2.nc` | gate、分层评价、OOD | P0 v2.2 已冻结；118,830/148,795 沿岸节点有 LME 标签 |

## 碳参数观测

| 数据 | 版本与范围 | 当前文件 | QC/用途 | 状态 |
|---|---|---|---|---|
| GLODAP | v2.2023；全球瓶样 | `C:\backup\phd\data\raw\ocads\glodapv22023\GLODAPv2.2023_Merged_Master_File.csv` | TA/DIC flag=2；旧表层锚 | 原始保留；旧 split 不再用于新正式实验 |
| CODAP-NA | v2026/version 2.2；1981–2024，446 航次 | `C:\backup\phd\data\raw\ocads\codap_na_v2026` | 北美 TA/DIC/pH/fCO2/氧/营养盐 | ZIP SHA256 `e5ddce21cdf257860a1f529eca9c1b8be7fbc3f0be792186f9427eb62a76624b` |
| CODAP+GLODAP 合并表层产品 | v1；北美 0–5 m，1972–2024 | `C:\backup\phd\data\processed\carbon\codap_glodap_na_surface5m_v1.nc` | 逐观测、来源/QC/重复组；训练筛 `is_primary=1` | 23,544 来源行；20,658 主记录；17,191 条 TA+DIC 均 good；SHA256 `edb2565decd8c4b75b62c8b4f5fc0d534d6fbaf6657aaddd4915d3e8cc06aed4` |
| Carter/ESPER TAest | ESPER LIAR v3 MAT | `D:\proj_personal\PhD\ESPER\ESPER_LIR_Files\LIR_files_TA_v3.mat` | 低保真先验/输入 | 不能作为 TA 真值；需在训练航次上估计 region/month bias |

合并产品输入哈希：CODAP CSV.gz 为 `ec233ae5d3070da9002a16fb1af7133168fee286d03868d3daa4e40fe94ed717`；GLODAP merged CSV 为 `cfae0bc5b38d9e2386f4a5d8569075f7785bce690b3d06ba82c9a7f51a61a820`。审计文件为同目录 `codap_glodap_na_surface5m_v1.audit.json`。

## Prepared 与模型缓存

| 产物 | 内容/规模 | 使用状态 | 关键限制 |
|---|---|---|---|
| `outputs/prepared_global_p32.nc` | 约 39.75 GB；1993–2026，408 月，148,795 coastal cells；SST、SSS_bg、ADT、wspd、pCO2air、SOCAT fCO2 | 保留，不因 CODAP 重建 | 2026 不完整；全量文件哈希待正式冻结 |
| `outputs/experiments/global_joint_cache_p32/joint_cache.npz` | SOCAT train/validation/旧 benchmark 752,668/239,396/48,934；GLODAP 10,359/2,824/784 | `weighted_global_v2` 基线 | 只含旧 GLODAP 和旧 split，不能用于新 CODAP 正式训练 |
| NACCOM caches | 东岸区域输入/标签 | 历史区域实验 | 结论不能直接外推全球 |
| Pacific-US caches | 西岸输入/标签 | 历史西岸实验 | `*_invalid_sal_path` 明确失效；只用修正目录 |
| North America compact joint cache v2.2 | SOCAT 522,322 个航次格月（到 2025）；碳观测 13,501 条 | P0 已冻结，Git 外目录 `C:\backup\phd\data\processed\recad_v2_2` | 按航次 split；外部航次统一封存 |
| inference availability v2.2 | 2025–2026 逐月逐沿岸节点 predictor 可用性 | 同上 `inference_availability_v2.2.nc` | 2025 全年严格可推理；2026 当前严格共同覆盖仅 1 月 |
| coastal graph / support map v2.2 | 148,795 节点、492,570 条四邻接有向边 | P0 已冻结，同上 | 图审计确认只连接水体节点；含最近碳支持距离 |

## 标签来源规则

- TA/DIC 主分析只接受 CODAP/GLODAP flag=2。
- flag 3/6 只进入预登记的敏感性分析。
- `measured`、`secondary-QC adjusted`、`temperature/pressure adjusted`、`CO2SYS-derived`、`LIAR/Carter-derived` 必须分开编码。
- SOCAT、CODAP、GLODAP 重复样本必须在 split 前按 cruise 和样本组锁定。
- 任何由 TA/DIC 通过 CO2SYS 计算的 fCO2/pH 不能被当作独立碳参数观测重复计权。
- 月平均状态经 CO2SYS 得到的是“月代表状态的化学一致诊断”，不自动等于瓶样碳参数的月平均。

## v2.2 冻结状态

P0 数据谱系、航次级 split、外部验证清单、LME/环境区、沿岸图、观测支持和 2025–2026 推理可用性均已完成，详见 `docs/p0_freeze_report_v2.2.md`。TA/DIC 在 2025–2026 可以作为模型预测/CO2SYS 派生输出存在，但不是观测标签。全球 Chl-a 仍未构建；若以后加入，必须提升 manifest 版本并重新预登记对应消融。

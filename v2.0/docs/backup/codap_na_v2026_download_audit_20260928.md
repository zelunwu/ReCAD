# CODAP-NA v2026 下载与初步审计（2026-09-28）

## 下载结果

- 官方数据集：CODAP-NA Version 2026，NCEI Accession 0315529，DOI `10.25921/h2ff-9d66`。
- NCEI 网页从中国网络无法稳定下载。已在 IPv6 服务器部署 Cloudflare WARP 本地 SOCKS 代理，并从 NCEI 的静态 FTP archive 下载 version 2.2。
- 本机目录：`C:\backup\phd\data\raw\ocads\codap_na_v2026`。
- 原始 ZIP：`SDIS_submission_CODAP_NA_V2026.zip`，41,598,471 bytes。
- ZIP SHA256：`e5ddce21cdf257860a1f529eca9c1b8be7fbc3f0be792186f9427eb62a76624b`。
- 已保留 NCEI metadata、readme、地图、经纬度清单和完整 `SHA256SUMS`。
- 已解压 `Output_file.csv.gz`、`Output_file.nc`、`Output_file.mat` 和营养盐修正表。

数据集包含 196,421 条观测、446 个航次，时间为 1981–2024。CSV/NetCDF 有 82 个字段，包括温盐、TA、DIC、pH、实测与计算 fCO2、氧和营养盐。

## 质量标志与使用规则

NetCDF 对 `TALK_flag` 和 `DIC_flag` 的定义为：

| flag | CODAP 含义 | 本项目用途 |
|---:|---|---|
| 2 | good | 主训练、验证和测试允许 |
| 3 | questionable but retained | 仅敏感性分析 |
| 6 | questionable | 仅敏感性分析 |
| 9 | missing or invalid | 排除 |

全水深 TA flag=2 有 94,552 条，DIC flag=2 有 99,673 条。正式碳参数锚点不得把 3/6 当成 good。

CODAP 元数据还说明：部分原位 pH/fCO2 以及计算变量在缺少实测 TA/DIC 时，会使用 LIARv2 盐度估算 TA 再运行 CO2SYS。因此联合模型必须记录每个碳变量是直接观测、温压订正还是 CO2SYS 派生，不能把后两者当作与 TA/DIC 独立的真值，否则会形成循环监督。

## 表层 0–5 m 严格 QC 统计

表层定义为 `Depth_meter <= 5 m`；深度缺失时使用 `CTDPRES_dbar <= 5`。下表只把 flag=2 计为合格 TA/DIC：

| 区域 | 表层行 | 航次 | TA good 行/航次 | DIC good 行/航次 | TA+DIC 均 good 行/航次 |
|---|---:|---:|---:|---:|---:|
| 北美边界窗 | 27,558 | 414 | 16,541 / 384 | 15,349 / 361 | 14,620 / 348 |
| 美国东岸 | 13,340 | 235 | 9,028 / 208 | 8,017 / 183 | 7,780 / 175 |
| 美国西岸 | 5,749 | 69 | 2,948 / 67 | 2,955 / 67 | 2,691 / 66 |
| Alaska | 3,155 | 62 | 2,082 / 58 | 1,996 / 58 | 1,948 / 55 |
| Gulf of Mexico | 6,912 | 49 | 6,219 / 47 | 6,040 / 49 | 5,955 / 47 |

区域窗会重叠，例如 Gulf 与东岸窗有交集，不能把各行直接相加。完整机器可读结果在数据目录的 `codap_na_v2026_surface5m_audit.json`，复现脚本是 `scripts/audit_codap_na_v2026.py`。

## 与现有 GLODAP 的关系

CODAP 有 446 个 EXPOCODE，当前 GLODAP 表层文件有 592 个。精确规范化 EXPOCODE 交集只有 74 个；CODAP 独有 372 个，GLODAP 独有 518 个。这说明 CODAP 会显著增加北美沿岸信息，但两个产品不能直接拼接：同航次需要优先按 EXPOCODE 去重，再用时间、经纬度、站位和深度匹配处理航次名不一致的重复观测。

## 下一步固定顺序

1. 生成逐样本 provenance 表，区分 measured、temperature/pressure adjusted 和 CO2SYS-derived。
2. 以 EXPOCODE 为第一层键、时空近邻为第二层键，对 CODAP 与 GLODAP 去重；同一观测优先保留 CODAP 沿岸 QC 信息，同时保留来源映射。
3. 冻结航次级拆分。任何同一航次或重复样本只能出现在一个 split；不再用随机行拆分。
4. 北美 TA 先比较区域线性、Carter/LIAR bias correction、层级 varying-coefficient 和生物过程残差模型。
5. DIC 先作为 `T + S + TA + fCO2` 的 inverse-CO2SYS 派生诊断。只有在直接 DIC 观测的航次外验证过关后，才考虑发布独立 DIC 产品。
6. 预留一批从未参与结构、超参或阈值选择的 CODAP 航次作为 locked test；其余训练集内部做 group-CV。最终独立验证继续使用与 SOCAT/CODAP/GLODAP 去重的外部航次或后续版本。


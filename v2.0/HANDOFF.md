# ReCAD v2.0 — Agent Handoff

> **当前权威入口（2026-10-08）**：数据版本与路径见 `docs/data_registry.md`；实验方法、结果和证据等级见 `docs/experiment_registry.md`；研究方向变化见 `docs/decision_log.md`；当前执行路线以 GitHub #1、#4、#5 和 `configs/frozen/product_scope_v2.3.json` 为准。下方早期章节保留工程历史，若与这些入口冲突，以当前权威入口为准。

> **2026-10-08 P2.0 范围冻结**：全球沿海 SSS/fCO2 是 P2 核心研究目标，但当前均未取得全球产品资格。SSS 正式状态为 `diagnostic_only`：#25 已使用一次 locked set，虽然通过 7/8 门槛，但 LME 17 失败，该集合永久不得再次称为 blind。fCO2 仅 Caribbean LME 12 为开发证据上的 `pass_regional`。MAB TA/DIC 为 `diagnostic_only`，SAB TA 为 `fail`；TA/DIC 只作为预算不超过 P2 10% 的北美可选扩展，失败不阻断核心产品。全球 atlas 只代表投影域，不代表全球观测验证。完整证据总账见 `docs/experiment_archive/p2_scope_reconciliation_v2.3/`。

> **2026-09-28 P0 v2.2 闭环**：数据轴保留 1993–2026；SSS/fCO2 核心期到 2025；TA/DIC 观测到 2024，但 TA 可前向预测、DIC 可由 inverse CO2SYS 推理到 2026。当前入口为 `configs/frozen/data_manifest_v2.2.json` 和 `docs/p0_freeze_report_v2.2.md`；v2.1 已替代。

> **2026-10-08 P1.5 闭环**：#24–#28 已完成。SSS 为带适用域标记的全球沿海候选；fCO2 只在 Caribbean Sea（LME 12）通过区域开发门槛；MAB TA 和派生 DIC 均为 `diagnostic_only`。#28 的完整 DIC 链 cruise/spatial/subregion/forward RMSE 为 49.61/58.90/51.06/39.53 µmol kg-1，2025 MAB 14,220 个 grid-month 因上游资格全部标 D 并 suppress。权威数值、图和限制见 `docs/experiment_registry.md` 与 `docs/experiment_archive/p1_dic_reliability_v2.2/`；locked TA/DIC 和外部 41 航次继续封存。

> **2026-09-05 更新**：模型选择进入独立验证优先的实验设计阶段，见
> [`docs/experiment_protocol.md`](docs/experiment_protocol.md)。尚未冻结正式 split 或运行模型比较。
> NACCOM 时间轴已修复；发现并修复 `build_standardized.py` 将 NOAA `average_unc`
> 误读为 xCO2 的列错误，全球/NACCOM xCO2 缓存已重建。区域 prepared/masks/stats
> 文件现已存在，但旧划分和归一化统计仅供工程检查，不能用于新独立验证实验。该段是 2026-09-05 历史状态；正式 v2.1 冻结已由上方 P0 更新替代。

> 交接文档(生成于 2026-09)。本文件给接手的 agent 提供完整上下文:
> 项目目标、数据现状、代码/脚本、环境坑、当前卡点、下一步行动。
> 读者应同时阅读 `docs/backup/design.md`、`docs/backup/loss_design.md`、`docs/data_download.md`。

---

## 0. 当前一句话

ReCAD v2.0 当前进入 P2：用真正全球的沿海观测证据分别建立并验证 SSS 偏差订正与 fCO2 重建；北美 TA 和由 `T+SSS+fCO2+TA` 经 inverse CO2SYS 派生的 DIC 是非阻断可选扩展。下一步依次执行 #37 文献与 v1.x 改进合同和 #38 全球观测缓存；任何全球 atlas 在通过目标专属验证前都只属于诊断投影。

---

## 1. 环境与机器

| 项 | 值 |
|----|----|
| 仓库根 | `D:\proj_personal\PhD\ReCAD` |
| 代码 | `v2.0/`(Python src-layout 包 `recad`) |
| venv | `v2.0\.venv`(python 3.x, torch 2.11+cu128, xarray, pandas, netCDF4, scipy, PyCO2SYS 1.8.3.4, dask, matplotlib, cartopy, shapely) |
| **数据卷** | `C:\backup\phd\data\raw\`(**数据不在仓库里**,D 盘只存代码) |
| GPU | **NVIDIA RTX 5070, 12.8 GB 显存** |
| 本机内存 | 48 GB |
| 远程服务器 | 临时 IPv6 下载服务器，密钥免密；连接信息只放本机 SSH 配置，不进入仓库 |
| 代理 | `127.0.0.1:7897`(Clash;下载走它;ssh/scp 不走) |

### ⚠️ 环境两大坑(必须知道)

1. **DSH 会话会周期性杀 python 长进程**(每 ~20 分钟,无日志、exit -1/被杀)。
   - **解法**:重活用 **Windows 计划任务**启动(`Register-ScheduledTask`,系统调度,活过清理)
     + **自愈 wrapper**(被杀自动重启)+ **断点续传**(按年/分片产出)。
   - 已验证:SSS/ADT/wspd 全靠"计划任务 + 逐年续传"完成。
   - **前台短任务(`Start-Process -RedirectStandardOutput ...`)可靠**;长任务(start 后 ≤15 分钟)可用,
     超过易被杀。**`2>&1 \| Select-Object` 会挂起,一律用 `*> 文件` 或 RedirectStandardOutput**。
2. **Smart App Control(SAC)曾拦截 python 模块加载**——已手动关闭
   (`HKLM\...\CI\Policy\VerifiedAndReputablePolicyState=0`),现在模块加载正常。

---

## 2. 数据现状(全部在 `C:\backup\phd\data\raw\`)

### 2.1 原始下载(≈750 GB)

| 数据集 | 位置 | 内容 | 状态 |
|--------|------|------|------|
| **SOCAT Coastal 散点** | `socat/SOCATv2026_Coastal.tsv` | **20,593,877 条逐观测**(含 fCO2rec/sal/temp, 1993-2026) | ✅ |
| GLODAP v2.2023 | `ocads/glodapv22023/GLODAPv2.2023_Merged_Master_File.csv` | 853 MB 全洋盆地合并(TA/DIC 全深度) | ✅ |
| GLODAP 5m 锚点 | `ocads/glodapv22023/glodap_surface5m_split.csv` | **14,930 表层(depth≤5m) TA/DIC**,已分割 | ✅ |
| CCMP v3.1 风 | `ccmp/`(Y{yyyy}/M{mm}/…V03.1_L4.nc) | 12,642 日文件(4×6h/天) | ✅ |
| OISST v2.1 | `sst/oisst-avhrr-v02r01.*.nc` | 12,289 日文件 | ✅ |
| GLORYS SSS | `sss/global/glorys12_so_daily_*.nc` | 34 年度文件(1/12°) | ✅ |
| CDS 海平面 | `sealevel/*.zip` | 34 zip(365 日 nc/年) | ✅ |
| xCO2air | `xco2air/noaa_gml_mbl/co2_mm_gl.txt` | 全球月序列 | ✅ |
| **注意** | `socat/SOCATv2026.tsv.zip`(全局 8.9GB)未处理(是航次清单非散点,见 §6) | | |

### 2.2 标准化(0.125° 月均)-> 管线输入 `v2.0/outputs/cache/`

| 文件 | 维度 | 内容 |
|------|------|------|
| sst.nc | (408, 1297, 2880) | OISST→月均,99% 覆盖 |
| sss.nc | (408, 1297, 2880) | GLORYS→月均,78% |
| adt.nc | (408, 1297, 2880) | CDS 海平面→月均,71%(0.25→0.125 最近邻) |
| wspd.nc | (408, 1297, 2880) | **CCMP RMS 风速**(sqrt(mean(ws²)),与 k∝u² 匹配) |
| xco2air.nc | (408, 1297, 2880) | NOAA 月均值广播 |
| fco2.nc | (408, 1297, 2880) | SOCAT binned 月均 fCO2(目标) |
| mask.nc | (1297, 2880) | 沿海掩膜 = 有观测格(31,151 NACCOM / 148,795 全球) |
| 时间 | 408 = 1993-01 .. 2026-12 | **注意 GLODAP 只到 2021(2022+ 无 TA/DIC 锚, masked loss 处理)** |

### 2.3 NACCOM 子缓存(用于快速验证)`v2.0/outputs/cache_naccom/`
同上但裁剪到 lon 260-320(480 列)、lat 10-65(441 行),各 ~330 MB,mask 31,151 沿海格。

---

## 3. 代码与脚本清单

### 3.1 关键模块(`v2.0/src/recad/`)

| 模块 | 功能 |
|------|------|
| `data/grid.py` | `DomainGrid` + `build_target_grid`(0.125° 全球) |
| `data/binning.py` | **散点→网格 {n, mean, std, std_err}**(`GriddedTarget`),`bin_tracks` |
| `data/tracks.py` | SOCAT 散点加载(**支持原生 TSV 表头定位**)、QC、`sample_predictors`、DataFrame roundtrip |
| `data/split.py` | **`split_points`(SOCAT:2004/05 整年 test + 航次级 val)**、网格 split |
| `train/joint_loss.py` | **联合反演损失**(§4),已单测通过 |
| `utils/chem.py` | PyCO2SYS pCO2air(**分块调用防 OOM**)、fCO2→pCO2 |
| `chem/esper_ta.py` | **ESPER-LIR( Carter)TA 移植**(方程16 纯SSS→TA),`from_mat` 读 LIR_files_TA_v3.mat |
| `model/st_transformer.py` | ST-Transformer(空间+时间 attention, mask_token 处理缺测,cell decoder) |
| `model/losses.py` | masked 损失(gaussian NLL / MSE / quantile) |
| `train/trainer.py` | 训练循环(深度集成、AMP、early stopping) |
| `data/pipeline.py` | `ingest_variable` / `build_prepared`(**已改 netCDF4 float32 直读防 OOM**) / `save_prepared` |

### 3.2 脚本(`v2.0/scripts/`)

| 脚本 | 用途 |
|------|------|
| `download_glodap.py` | GLODAP v2.2023 下载(glodap.info / NCEI, csv.zip) |
| `extract_glodap_anchors.py` | 提取表层 TA/DIC 锚点(带 expocode,可按深度/年份筛) |
| `split_glodap.py` | GLODAP 锚 train/val/test(2004/05 test + expocode 分组) |
| `plot_glodap_anchors.py` | 锚点覆盖图(近海覆盖分析) |
| `build_standardized.py` | 全球 cache 聚合(SST/SSS/ADT/wspd…,支持 `--piece k,n` 并行) |
| `build_wspd_year.py` | **wspd 逐年构建(断点续传,抗杀)** |
| `wspd_all.py` | wspd 自愈 wrapper(逐年直到完成→合并 wspd.nc) |
| `slice_cache.py` | 全球 cache → 子区域(如 NACCOM) |
| `merge_standardized_pieces.py` | 合并 `name_piece{k}.nc` → `name.nc` |
| `run_mvp.py` | 早期 MVP(MLP 占位 + 气候态预测因子,**已过时,仅参考 pipeline 骨架**) |
| `average_daily_to_monthly.py` | 日→月平均(GLORYRS 用) |

---

## 4. 联合反演损失设计(`docs/backup/loss_design.md` — 详细)

```
模型输出:  fCO2̂, SSŜ, TÂ, DIĈ
预测输入:  SST, SSS, ADT, pCO2air, wspd + lon/lat/月份 + (进阶)TA_ESPER先验

L = λ_fCO2·mean((fCO2̂−fCO2_SOCAT)²/σ_fCO2²)      # SOCAT ~2000万点
  + λ_SSS ·mean((SSŜ−sal_SOCAT)²/σ_SSS²)          # SOCAT sal
  + λ_TA  ·mean((TÂ−TA_GLODAP)²/σ_TA²,  mask)     # GLODAP/CODAP 锚(稀疏,masked)
  + λ_DIC ·mean((DIĈ−DIC_GLODAP)²/σ_DIC², mask)   # 同上
  + λ_chem·mean((CO2SYS(T,S,TÂ,DIĈ)−fCO2_SOCAT)²/σ_chem²)  # 化学闭合,全样本
  + λ_smooth·Σw(x)(∇²TÂ)²+(∇²DIĈ)²+(∇²SSŜ)²     # 空间二阶平滑,沿岸加权

默认: λ={fCO2:1, SSS:1, TA:5, DIC:5, chem:1, smooth:0.05}
      σ ={fCO2:5, SSS:0.1, TA:8, DIC:8, chem:5}
```

**要点**(设计定稿):
- **不做时间平滑**(monthly anomaly 的月际跳变是真实信号);可选 AR(1) 先验默认关
- **含 ESPER TA 先验**:`TA_Carter(SSS,SST,lon,lat)` 可作为模型输入通道或 δTA 残差锚
  (ESPER 方程16 已验证:ΔTA/ΔS≈60-70 µmol/kg/PSU,与 Lee 2006 一致;**近岸未验证,当先验非产品**)
- **历史状态（已过期）**：当时 CODAP 下载受 NCEI 503 阻塞。CODAP-NA V2026 现已发布；最新动作见 2026-09-28 节。

---

## 5. 历史卡点（2026-09 初；请以文末 2026-09-28 节为准）

### 5.1 主要卡点:**全量 1/8° 训练的内存/规模问题**

| 问题 | 细节 |
|------|------|
| **全量 `recad preprocess` OOM** | 6 变量全载入 34GB + PyCO2SYS float64 11GB > 48GB |
| 已修复 | `build_prepared` 改 netCDF4 float32 直读、`chem.py` 分块(见 §3) |
| **仍未验证** | 全量 preprocess 还没成功跑完(最后一次停在写盘阶段被手动 kill) |
| **当前策略** | 用 **NACCOM 区域**先验证全流程(NACCOM 刚裁剪好),**全量生产运行需要分块或更高内存机** |

### 5.2 立即要做的(下一步)

1. **修复 NACCOM 配置的时间维对齐**,然后跑 `recad preprocess`:
   - 报错:`ValueError: cannot reshape array of size 86365440 into shape (29,12,441,480)`
   - 原因:`naccom_1over8.yaml` 未写 `data` 段 → 继承 base 的 `year_max: 2021`(348 月),
     而 cache 是 408 月(1993–2026)。
   - **修法**:在 `naccom_1over8.yaml` 加:
     ```yaml
     data:
       root: C:/backup/phd/data/raw
       year_min: 1993
       year_max: 2026
     ```
     若想用全量 408 时间轴;或改 slice 成 348。**推荐 2026(与 cache 一致)。**
2. preprocess 成功后 → `recad train --config configs/naccom_1over8.yaml --prepared outputs/prepared_naccom.nc --masks outputs/masks_naccom.nc`
   - **阶段①小配置**(默认 naccom 已 embed 64/2 层,~1h);**RTX 5070 12.8GB,cuda 可用**
3. `recad predict` → `recad validate` → 三段 R²/RMSE(train/val/test 2004-05)

### 5.3 其他已知待办

- **历史记录**：CODAP-NA V2021 下载曾受 NCEI 503 阻塞。CODAP-NA V2026 已于 2026 年发布，
  现应直接获取 V2026（NCEI Accession 0315529），不再继续旧 watchdog 路线。
- **全量训练**:NACCOM 验证后,全量需处理内存(建议:分 lat 块训练 或 提升到 ~64GB 内存机;
  数据集 30GB float32 是硬门槛)。
- **MODIS Chl-a**(2002+ 生物通道)未下载(用户计划自己处理)。
- **CODAP vs GLODAP 重叠**:陆架外缘过渡带去重策略(expocode + 距陆 <100km 以 CODAP 优先)——待 CODAP 到位实现。
- **ESPER 未来**:DIC/pH 输出未移植(需 Canth 时间调整);不确定性网格(UncGrid)未移植。

---

## 6. 有用的历史细节 / 避坑

- **SOCATv2026.tsv.zip / _Coastal.tsv.zip** 是"航次清单"(Per-cruise metadata),
  **不是逐观测数据**;逐观测在 `SOCATv2026_Coastal.tsv`(已解压,2050万行,表头在第 6805 行,
  列名含 `fCO2rec [uatm]`、`sal`、`SST [deg.C]`、`dist_to_land [km]`)。
- `load_tracks` 自动定位 TSV 表头(**找含 Expocode+fCO2rec 的行**)并映射列名。
- **CCMP 变量名**:日文件用 `ws`,月均文件用 `w`——聚合时按 `"ws" if "ws" in vars else "w"`,并排除 `monthly_mean` 文件。
- **GLODAP 无近似**:TA/DIC 用 `G2talk`/`G2tco2`,质量旗标 `G2talkf`/`G2tco2f ≤ 2`。
- ESRPER `LIR_files_TA_v3.mat` 结构:`Cs[50225,6,16]`(节点×6系数×16方程)、`GridCoords[50225,4]`(lon,lat,depth,date)、
  `AAIndsCs`(大西洋/北极分段)、`Polys`(区域多边形,用 matplotlib Path 判含)。`scipy.loadmat` 读 Polys 字段直接用 `p[f]`(非 `p[f][0][0]`)。
- **数据不入 git**:根 `.gitignore` 有 `/data/`;数据全在 `C:\backup\phd\data\`。脚本入仓库。

---

## 7. 复现一个完整小跑(最快路径验证)

```powershell
# 1) NACCOM 预缓存已就绪 outputs/cache_naccom/(含 mask)
# 2) 修 naccom_1over8.yaml 加 data.year_max=2026(见 §5.2)
# 3) 计划任务方式跑 preprocess(或 Start-Process 前台短任务):
python -m recad preprocess --config configs/naccom_1over8.yaml --cache outputs/cache_naccom --prepared outputs/prepared_naccom.nc --masks outputs/masks_naccom.nc
# 4) 训练(阶段①):
python -m recad train --config configs/naccom_1over8.yaml --prepared outputs/prepared_naccom.nc --masks outputs/masks_naccom.nc --stats outputs/feature_stats_naccom.json
# 5) 验证:
python -m recad validate --config configs/naccom_1over8.yaml --prepared outputs/prepared_naccom.nc --masks outputs/masks_naccom.nc
```

---

## 8. 关键命令速查

```powershell
# 计划任务跑长任务(模板)
$action = New-ScheduledTaskAction -Execute "venv\Scripts\python.exe" -Argument "-X utf8 -u `"C:\path\script.py`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
Register-ScheduledTask -TaskName XXX -Action $action -Trigger $trigger -Force

# 合并分片
python scripts/merge_standardized_pieces.py --cache outputs/cache --names sss,adt,wspd --years 1993,2026

# GLODAP 锚点提取+分割
python scripts/extract_glodap_anchors.py --csv <merged.csv> --out <anchors.csv> --max-depth 5
python scripts/split_glodap.py --csv <anchors.csv> --out <split.csv> --test-years 2004,2005

# ESPER TA 先验
python -c "from recad.chem.esper_ta import ESPER_LIR_TA; e=ESPER_LIR_TA.from_mat(r'D:\proj_personal\PhD\ESPER\ESPER_LIR_Files\LIR_files_TA_v3.mat'); print(e.estimate([280.], [25.], [0.], [36.], [28.], equation=16))"
```

---

**给接手者的第一建议**:先完成 §5.2(修 naccom 配置 → preprocess → 训练 → validate),
拿到一个能跑的 1/8° NACCOM 基准;再决定全量(内存)与联合反演(TA/DIC/SOCAT 双源)的推进顺序。
祝顺利。

## 2026-09-28：CODAP-NA v2026 已下载并完成严格 QC 初审

NCEI 网页在中国网络不稳定，现已通过新 IPv6 服务器上的 Cloudflare WARP SOCKS 代理，从 NCEI 静态 FTP archive 下载 Version 2.2。完整数据已复制并解压到 `C:\backup\phd\data\raw\ocads\codap_na_v2026`。主 ZIP SHA256 为 `e5ddce21cdf257860a1f529eca9c1b8be7fbc3f0be792186f9427eb62a76624b`。

数据共有 196,421 条、446 航次、1981–2024。表层 0–5 m 严格按 flag=2：美国东岸 TA/DIC 均 good 为 7,780 条、175 航次；西岸 2,691 条、66 航次；Alaska 1,948 条、55 航次；Gulf 5,955 条、47 航次。CODAP 与当前 GLODAP 表层文件精确 EXPOCODE 重叠 74 航次，CODAP 独有 372 航次，不能直接拼接。

详细审计和下一步见 `docs/backup/codap_na_v2026_download_audit_20260928.md`；复现脚本为 `scripts/audit_codap_na_v2026.py`。主分析只接收 TA/DIC flag=2，flag 3/6 仅用于敏感性分析。必须区分直接观测与 CO2SYS/LIARv2 派生变量，防止循环监督。下一步先做 CODAP+GLODAP 的 provenance 与航次/时空去重，再冻结 group-CV 和 locked cruise test。

## 2026-09-28：CODAP + GLODAP 表层逐观测文件已构建

新增 `scripts/build_codap_glodap_surface_nc.py`，从 CODAP-NA v2026 和 GLODAPv2.2023 原始文件生成可审计的 observation-indexed NetCDF。输出位于仓库外的 `C:\backup\phd\data\processed\carbon\codap_glodap_na_surface5m_v1.nc`，配套审计为同名 `.audit.json`，数据不会进入 Git。

口径为北美 0–75°N、180–45°W，表层 0–5 m，保留 TA 或 DIC 至少一个 flag=2 的记录。CODAP 保留 17,270 条，GLODAP 保留 6,274 条；23,544 条来源记录中识别出 2,886 对跨产品重复观测，均保留 provenance，但只有一条 `is_primary=1`。去重后主记录 20,658 条，good TA 19,356 条、good DIC 18,493 条、二者均 good 17,191 条，577 个 EXPOCODE。重复匹配中 2,490 对为相同规范化 EXPOCODE，396 对为日期、位置、深度和碳参数均高度一致的航次别名/占位号。

该 NetCDF 仍是逐瓶/逐样本表，不是月平均格点。下一步从 `is_primary=1` 记录生成航次级 split manifest，再与现有 `prepared_global_p32.nc` 匹配生成北美 compact joint cache。

## 2026-09-28：沿岸空间非平稳性结构决策

后续文献追踪必须并行覆盖最新地球科学方法与最新计算机科学方法。硬生物地球化学分区分别训练会产生边界跳变；只给统一模型加入 lon/lat 又不能防止高观测密度区域支配权重。主候选确定为“全局主干 + 软门控区域残差 experts”：环境状态决定重叠的 top-2 gate，低数据区域通过收缩退回全局模型，SSS/fCO2/TA 允许不同 task gate，DIC 仍由 CO2SYS 派生。

训练从数据端同时修正偏置：按 region→cruise→month 分层采样，并比较区域宏平均、worst-group/CVaR 与普通 pooled loss。空间邻域采用沿水体连通的 coastal graph/attention，避免隔陆近邻；连续性通过软 gate 和图正则实现，但在锋面、河口和地形屏障处降低平滑。P1/P2 增加统一模型、区域平衡、区域残差 adapter、soft MoE、soft MoE+coastal graph 的受控对照，并专门报告整 LME 留出和边界带跳变。

## 2026-09-28：GitHub Roadmap 与实验总账

GitHub 父 Roadmap 为 issue #1：`https://github.com/zelunwu/ReCAD/issues/1`。GitHub sub-issues 为 #2 P0 数据/验证冻结、#3 P1 单任务可行性、#4 P2 空间非平稳与结构化联合模型、#5 P3 锁定盲测。阶段内部实验保留为各子 Issue 的 checklist，避免建立大量失去上下文的小 Issue。

新增 `docs/experiment_registry.md` 作为所有实验的统一索引，历史实验按 A/B/C/X 证据等级登记；新增 `configs/experiment_record_template.yaml`。今后正式实验必须在训练前复制并填写模板，提交元数据记录，再运行训练。大文件、checkpoint、预测和完整 metrics 仍只存 `outputs/experiments/<id>/` 或外部数据盘，不进入 Git。

## 2026-09-06：受控模型实验已完成

最新入口：`docs/backup/controlled_experiments_20260906.md`。已完成 2 个 tiny 诊断、8 个 train/dev 条件、2 个 5,000 步延长实验，总计 26,000 次更新。结果在 `outputs/experiments/controlled_20260906/`，有协议/数据/代码哈希、best 与可续训 last、完整预测、指标及图。

最佳开发候选：`st96_mse_lr1e3/best.pt`，选中第 2,750 步，train RMSE=20.736、真 R2=0.833，dev RMSE=27.326、真 R2=0.653。深解码器训练到 5,000 步：train=19.782、dev=27.643。训练集合 158,132、dev 38,949，全量评分覆盖 100%。本轮没有重新评分 test 或空间挑战，更没有外部独立验证结论。

修正 bootstrap 未抽年份仍被使用的问题；加入稀疏监督查询解码和 SDPA attention，23 项相关测试通过。实际 patch8 已在新 runner 显式校验；原 YAML patch16 与 prepared patch8 不一致的问题已记录，旧实验未改写。标准 CLI 的统计量持久化、独立地理 coast mask、逐变量 token coverage 仍需后续处理；本轮冻结其旧口径以隔离模型/优化因素。

下一步优先复现最优候选、多种子与同数据树模型基线，不要因 training 分数下降就认定外推改善。不要改写 `controlled_20260906/protocol.json` 后混入新条件；新条件建立新版本目录。

## 2026-09-06：MODIS Chl-a 已标准化并完成独立测试

叶绿素原始目录实际为 `E:\T7_4T\modis\mapped_monthly\chlor_a`。`scripts/build_naccom_chla.py` 已按现有 NACCOM 数据处理口径重网格到 0.125°，生成 `outputs/cache_naccom/chla.nc`：Aqua/MODIS 2002-07--2020-12，物理单位 mg m-3，最近邻重网格、范围 (0,100] QC；模型输入用 `log10(chla)` 并保留缺失标记。

新实验严格隔离在 `outputs/experiments/chla_ablation_20260906/`，不改写之前的 `controlled_20260906`。两种条件在完全相同的、同时有 SOCAT fCO2 与 Chl-a 的 112,801 train / 31,923 dev / 12,242 个当时留出点比较，三种子、各 5,000 更新。开发集三种子 RMSE 中位数由 25.743 降至 24.127（-6.28%）。当时在预先固定规则下评分的 2004--2005 集得到：三成员集成 RMSE 23.115 → 21.099，MAE 14.796 → 13.456，R² 0.618 → 0.682。该集合此后已被反复查看，当前只能称历史 benchmark。详情见 `docs/backup/chla_ablation_20260906.md`，图为 `outputs/experiments/chla_ablation_20260906/development_rmse.png`。

注意：该结论只覆盖 MODIS 存在的 2002-07--2020-12；生产重建必须为 1993--2002 与 2021+ 保留无 Chl-a 的模型或建立有原则的缺失方案，不能填零后假装全时段可用。

## 2026-09-08：R0–R4 稀疏碳参数实验完成

最新报告：`outputs/experiments/joint_r0_r4_20260908/training_report_zh.md`。五个条件均完成 5 折 expocode/空间分组 CV（每折 2000 step），前两名完成 3 种子最终训练。排名为 R0 共享头 0.4529、R2 宽化学头+3:1 采样 0.4710、R4 Huber+弱化学约束 0.4764、R1 宽化学头 0.4809、R3 TAest dropout 0.4841。

R0 的 internal test RMSE 为 SSS 0.860、fCO2 25.409、DIC 40.983、TA 47.298；2004–2005 benchmark 为 1.129、23.223、36.895、40.405。它缩小了 train–test 间隙，但 benchmark DIC/TA 和 exact CO2SYS 闭合均未超过旧 B2。R2 只让 benchmark TA 从 38.635 微降至 38.497，其他关键结果退化。结论是当前瓶颈为航次/水团覆盖与分布偏移，不是网络容量；暂停继续堆大模型，旧 B2 保持当前生产候选。

2004–2005 已被多轮查看，只能作为 benchmark，不能再称完全独立验证。下一步必须先锁定一个从未用于模型决策的外部数据源、航次或区域，再比较旧 B2、R0 和低频背景+残差的 DIC/TA 模型。

## 2026-09-09：结构化 CO2SYS 与美国东西海岸实验完成

最新报告：`outputs/experiments/us_coasts_carbonate_report_20260909/report_zh.md`。碳酸盐模型改为只独立预测 fCO2 与 TA，DIC 由 `T+S+TA+fCO2 -> DIC` 可微逆 CO2SYS 解码器生成；逆解码器宽域 RMSE 0.552 µmol/kg。模型报告以 SSS/fCO2 为主，TA 做美国东西岸专题，DIC 为派生诊断。

东岸 C1–C3 完成 5 折 + 前两名 3 种子，C2 概率模型 CV 最佳 0.3807±0.0101。internal test SSS/fCO2/TA/DIC RMSE 为 0.856/25.04/48.06/39.06，但 2004–2005 benchmark 为 1.198/23.74/46.06/42.39，未超过旧 B2，因此旧 B2 仍为东岸生产候选。东岸 C2 TA 90% 区间在 internal test 只覆盖 71.7%，不确定性偏小。

新建 Pacific-US 225–250°E、20–60°N 数据域：408 月、11,149 沿岸格点，MODIS 有效 6,933,072 网格月；从 4 GB SOCAT TSV 流式聚合 2,280,787 条现场盐度观测为 62,489 网格月。西岸 C1–C3 完成相同协议，C1 CV 最佳 0.5483±0.0345。internal test SSS/fCO2/TA/DIC 为 0.748/42.35/38.97/52.49，但五个航次折 TA RMSE 为 99.3/126.5/114.1/74.6/88.7，随机 test 明显偏容易。2004–2005 西岸 TA 仅 10 点（美国本土窗 5 点），不能作为独立结论；西岸当前不得进入生产。

西岸初次缓存曾错误沿用东岸 `SAL` 路径，结果已整体移动到 `*_invalid_sal_path`，未被最终结果复用。通用训练代码同时修复了“batch 内现场 SSS 全缺失时空均值产生 NaN”的边界条件。

## 2026-09-28：全球数据/实验审计与产品路线重定

最新决策文档：`docs/current_status_and_roadmap_20260928.md`。产品拆成两层：全球沿海主产品只做 SSS/fCO2；北美沿岸扩展做 TA，并在 TA 适用域内通过 inverse CO2SYS 派生 DIC。全球无 TA 支持区域不发布 DIC。

9 月 24 日已完成全球 prepared 和纠正后的联合缓存：`prepared_global_p32.nc` 约 39.75 GB，1993–2026 共 408 月、148,795 个 coastal-mask 格；SOCAT train/validation/2004–2005 benchmark 为 752,668/239,396/48,934，现场盐度聚合为 890,581 格月；GLODAP 表层聚合锚为 10,359/2,824/784。2026 年只有 36 个 SOCAT 格月，正式版本应截止 2025 或最后完整年份。全球 Chl-a 尚未构建。

`weighted_global_v2` 是修复 SSS 目标泄漏后的有效全球基线：SSS validation/旧 benchmark RMSE 1.156/1.081 PSU；fCO2 为 60.45/53.20 µatm，旧 benchmark skill −0.138，全球统一 fCO2 模型未过关。高达 0.986 的全球 SSS R²受极端低盐水团扩大方差影响；美国窗口 skill 仅约 0.39–0.46，因此 SSS 应称为 GLORYS 沿岸偏差订正，并按河口/盐度段报告。

2004–2005 已反复用于模型选择，只保留为 v1.1 历史 benchmark。下一轮 fCO2 采用三类外层评价：整航次留出、整 LME/空间区留出、forward-chaining 时间外推；最终外部验证使用与 SOCAT 去重后的 OCADS mooring/OceanSITES/Saildrone 或冻结模型之后的新 SOCAT 版本。产品主指标按 grid-month、cruise 和 region 三种等权口径报告。

NOAA/NCEI 已发布 CODAP-NA V2026（1981–2024，446 航次、32,250 剖面，含 TA/DIC/pH/fCO2/氧/营养盐）。下一步 TA/DIC 的首要工作是下载 CODAP、按 EXPOCODE+时空与 GLODAP 去重，然后比较区域线性、Carter bias correction、层级 varying-coefficient TA–SSS 和生物过程残差模型。最新 `smoke_sel` 因 `run_weighted_carbonate.py` 缺少 `selection_score` 定义而失败；任何正式重训前先修复并加测试。

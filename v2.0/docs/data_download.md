# 数据下载指南（ReCAD v2.0）

所有数据放在**中央数据卷 `C:/backup/phd/data/raw/<source>/`**（2026-08-28 从
`v2.0/data/raw/` 迁移；D 盘仅存代码，数据不再放仓库里）。该目录不在 git
仓库内，**任何科学数据都不会进入 git**；可复现性由
`C:/backup/phd/data/raw/MANIFEST.md`（`recad download` 自动生成，记录来源、
URL、文件、校验和）保证。配置 `configs/*.yaml` 的 `data.root` 与下载脚本
已统一指向新位置。

> 规模提醒：全球 1/8° 的全套数据（SOCAT/OISST/GLORYS/SSH/CCMP）总计
> 数十 GB。本指南给出的都是**最小必要**下载方案（先按 NACCOM 窗口或
> 概要变量下载验证流程，再扩到全球）。

---

## 0. 一键下载（开放源）

```bash
recad download --list                          # 查看清单
recad download --dry-run                       # 只规划不下载
recad download                                 # 下载所有开放源（默认 NACCOM 窗口）
recad download --only socat --region -100,-40,10,65 --years 1993,2021
recad download --doc                           # 打印各源说明
```

开放源（无需注册）：`xco2air`、`socat`、`gshhg`。
注册源（需账号，见下）：`sst`、`sss`、`adt`、`wspd`、`bathymetry`。

---

## 1. xCO2air — NOAA GML 大气 CO2（开放，约 20 KB）

- 最新版本：NOAA GML Marine Boundary Layer 月度参考产品（v1.1 同源）。
  提供两种文件：
  - **MBL 纬带平均 surface 文件**（v1.1 用的 `Global_90S90N_*` 系列）：
    <https://gml.noaa.gov/ccgg/mbl/>
  - **全球月均序列**（本工具默认下载）：
    <https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.txt>
- 命令：`recad download --only xco2air`
- 落地：`C:/backup/phd/data/raw/xco2air/noaa_gml_mbl/co2_mm_gl.txt`（附 `.sha256`）
- 使用：`recad ingest` 通过 PyCO2SYS 在逐格 SST/SSS 下转 pCO2air。

## 2. SOCAT 海岸 fCO2（开放，按窗口从 ~10 MB 到数 GB）

- 最新版本：**SOCAT v2026** 0.25° 海岸月度网格产品（PMEL ERDDAP，
  `SOCAT_v2026_qrtrdeg_gridded_coast_monthly`；正式文档数据库为
  SOCAT v2025，NCEI Accession 0304549）。
- 本工具自动：读取 ERDDAP `.dds` 元数据 → 取坐标轴 → 按经纬度/年份切
  片 → 下载单变量 NetCDF（断点续传）。
- 命令：
  ```bash
  recad download --only socat --region=-100,-40,10,65 --years=1993,2021   # NACCOM 窗口
  ```
- 落地：`C:/backup/phd/data/raw/socat/SOCAT_v2026_coast_monthly_lon.._lat.._1993-2021.nc`
- 手工方式（浏览器）：打开
  <https://data.pmel.noaa.gov/socat/erddap/griddap/SOCAT_v2026_qrtrdeg_gridded_coast_monthly.html>
  选择变量与窗口导出 `.nc`。

### 2b. SOCAT 原始散点观测（tracks）——训练推荐（本仓库默认路线）

网格化产品把每个 0.25° 格内的观测**平均**掉了，丢失船测航迹的原位空间
信息（河口、陆架锋、近岸梯度）。**点级（scatter）训练**保留每个观测的
原始经纬度，预测因子在观测位置采样——这正是 1/8° 海岸重建需要的精度。

- 数据源：PMEL ERDDAP **tabledap** `socat_v2026_decimated`（1 分钟抽稀，
  建模标准）或 `socat_v2026_fulldata`（全部观测，体量大很多）。
- 命令：
  ```bash
  # NACCOM 窗口 1993-2021（decimated；约几十 MB，视航次密度）
  recad download --only socat_tracks --region=-100,-40,10,65 --years=1993,2021
  # 全球窗口（很大；建议按区域/年份分批下）
  recad download --only socat_tracks --region=0,360,-78,84 --years=1993,2010
  # 全部观测（不含 1 分钟抽稀；数据量约为 decimated 的 3-5 倍）
  recad download --only socat_tracks --kind=fulldata --region=-100,-40,10,65 --years=1993,2021
  ```
- 落地：`C:/backup/phd/data/raw/socat_tracks/SOCAT_v2026_decimated_lon.._lat.._1993-2021.nc`
  每行一个观测：`time, latitude, longitude, fCO2_recommended,
  WOCE_CO2_water(QC旗标), dataset_name, sal, temp`。
- 读取/QC/预测因子采样（`recad.data.tracks`）：
  ```python
  from recad.data.tracks import load_tracks, qc_tracks, prepare_point_table
  from recad.data.pipeline import load_prepared
  tr = qc_tracks(load_tracks("C:/backup/phd/data/raw/socat_tracks/*.nc"), qc_flag_max=2)
  tab = prepare_point_table(tr, load_prepared("outputs/prepared.nc"))
  # tab.predictors: [n, 5] = 每个观测处采样的 sst/sss/adt/pco2air/wspd
  ```
- QC 说明：`WOCE_CO2_water <= 2` 视为可用（SOCAT 约定）；fCO2 范围与
  3σ 规则沿用 v1.1（`recad.constants.QC_RANGES`）；**划分训练/验证/测试**
  必须在散点层面按时间或船次进行（点级训练的下一步，见 docs/backup/design.md）。
- 整库（可选）：`files/socat_v2026_fulldata/`、`files/socat_v2026_decimated/`
  目录下有逐船次 CSV，可全量镜像；
  文档数据库 SOCAT v2025：<https://catalog.data.gov/dataset/surface-ocean-co2-atlas-database-version-2025-socatv2025-ncei-accession-0304549>。

## 3. GSHHG 海岸线（开放，约 55 MB）

- 版本：**GSHHG 2.3.7**，Zenodo record [7007502](https://zenodo.org/records/7007502)
  （工具经 Zenodo API 解析最新文件链接）。
- 命令：`recad download --only gshhg`
- 落地：`C:/backup/phd/data/raw/gshhg/gshhg-shp-2.3.7.zip` + 解压目录
- 使用：v2.0 `coastal_mask.method: distance` 可复现海岸掩膜的几何底图
  （`recad ingest` 时把掩膜栅格化成 `C:/backup/phd/data/raw/mask.nc`；或直接下载社区
  预生成掩膜置于 `C:/backup/phd/data/raw/mask.nc`）。

## 4. SST — OISST v2.1（0.25°，日平均，**免账号直链**）

- 版本：**OISST v2.1**（NCEI 开放归档，无需注册；当前主产品，无 v3）。
- 官方目录（月为单位组织）：
  `https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/YYYYMM/`
  每日文件 `oisst-avhrr-v02r01.YYYYMMDD.nc`（sst/err/ice/anom 四变量，
  0.25° 全球，**约 1.7 MB/天** → 一年约 620 MB）。
- 工具下载（推荐，断点续传 + 校验）：
  ```bash
  # 单个验证月
  recad download --only sst --years=1993,1993 --months=01
  # 整年（1993 全年 = 12 个月目录，约 620 MB）
  recad download --only sst --years=1993,1993
  # 多年（NACCOM 重建 1993-2021 全量约 18 GB，建议分批/后台）
  recad download --only sst --years=1993,2000
  ```
- wget 直抓（一个月；用户自选）：
  ```bash
  # 方式 A：整目录镜像（每月目录）
  wget -r -np -nH --cut-dirs=5 -A 'oisst-avhrr-v02r01.*.nc' \
       https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/199301/
  # 方式 B：curl 循环（更直观）
  for d in $(seq -w 1 31); do
    curl -O "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/199301/oisst-avhrr-v02r01.199301${d}.nc"
  done
  ```
- 落地：`C:/backup/phd/data/raw/sst/*.nc`（保持原名即可；`recad ingest` 自动日→月平均并重采样）。
- 注意区分 `access/avhrr/`（AVHRR-only，与 v1.1 相同）与 `access/avhrr-asc/`
  （加 ASCAT，产品页选择时注意）；本仓库默认前者。
- 替代（可选）：MUR 1 km（2002+）、ESA CCI/C3S 0.05°（见 `docs/data_sources.md` §4）。

## 5. SSS — GLORYS12v1（CMEMS，需注册；用官方 Copernicus Marine Toolbox）

- 产品：CMEMS `GLOBAL_MULTIYEAR_PHY_001_030`（GLORYS12V1，1/12°，
  1993→今）。CMEMS 需免费注册：<https://data.marine.copernicus.eu>
- **Python API 已装好**（venv 内 `copernicusmarine` 2.4.x；API 文档：
  <https://help.marine.copernicus.eu/en/articles/8283072-copernicus-marine-toolbox-api-subset>）。
- 第 1 步：登录一次（凭据存本机用户目录 `~/.copernicusmarine/`，之后无需重复）：
  ```bash
  copernicusmarine login
  # 或环境变量 COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD
  ```
- 第 2 步：子集下载表层盐度（全局 1/8° 窗口，日数据）——**推荐用封装脚本**
  （按年分块、断点续传、跳过已存在年份；相对路径一律从仓库根解析，
  与调用时的工作目录无关，输出固定到 `C:/backup/phd/data/raw/sss/{前缀}_{年}.nc`）：
  ```bash
  python scripts/download_glorys_sss.py --region=0,360,-78,84 --years=1993,2026
  # 只下全球窗口的表层 so：区域 0–360 / −78..84，每年度文件约 5.7 GB
  ```
  NACCOM 窗口的旧用法（仅当只需北美东海岸时）：
  ```bash
  python scripts/download_glorys_sss.py --years=1993,2025 --region=-100,-40,10,65
  ```
  ⚠️ 若按本文上面几节手动调用 `copernicusmarine subset`，`--output-directory`
  在后台脚本里是相对路径、会随 cwd 变化——务必用绝对路径，否则数据会散落到
  仓库根目录的 `data/` 下（根 `.gitignore` 已兜底，但目录会乱）。
  等价的手工单次调用：
  ```bash
  copernicusmarine subset --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m ^
    --variable so --start-datetime 1993-01-01 --end-datetime 2026-12-31 ^
    --minimum-longitude 0 --maximum-longitude 360 ^
    --minimum-latitude -78 --maximum-latitude 84 ^
    --minimum-depth 0.49402499198913574 --maximum-depth 0.49402499198913574 ^
    --output-directory C:/backup/phd/data/raw/sss --output-filename glorys12_so_daily_1993
  ```
  （注意 `--minimum-depth` 必须用数据集精确坐标 `0.49402499198913574`，
  传近似值 `0.494` 会报 "Coordinates out of dataset bounds"；
  数据集 id 如有出入，用 `copernicusmarine describe --include-datasets`
  搜 `GLOBAL_MULTIYEAR_PHY_001_030` 确认。）
- 第 3 步：日 → 月平均（`recad ingest` 需要月尺度时间轴）：
  ```bash
  python scripts/average_daily_to_monthly.py ^
      --glob "C:/backup/phd/data/raw/sss/glorys12_so_daily*.nc" ^
      --out C:/backup/phd/data/raw/sss/glorys12_so_monthly.nc
  ```
- 备选方案对比与误差策略见 `docs/data_sources.md` §3b；无 CMEMS 时回退：
  SOCAT gridded 自带 `coast_salinity_ave_weighted`（已随 socat 下载），
  或 SMAP V5.0（2015+，近岸 ~100 km 内不可靠，仅作独立校验）。

## 6. ADT/SSH — CMEMS SEALEVEL_GLO_PHY_L4_MY_008_047（需注册）

- 与 §5 同一 CMEMS 账号。产品仍为 0.25° 日平均延迟产品（全球无 1/8°
  L4 网格，见 `docs/data_sources.md` §5）。
- 下载 `adt`（优先，保留平均动力地形）月度平均 → `C:/backup/phd/data/raw/adt/*.nc`。

### 6b. 海平面/ADT — C3S satellite-sea-level-global（CDS，需账号）

- 产品：C3S **satellite-sea-level-global**（多卫星融合 L4，0.25°，日平均；
  版本 vdt2024）。CDS 单请求成本上限 372，一年日数据恰好 ~372，
  因此**必须按年切分**：34 年全量请求成本 12648 会被拒绝。
- 下载脚本：`python scripts/download_sealevel_cds.py`（逐年循环、断点续传、
  zip 校验；依赖 `~/.cdsapirc`，配置方法见
  [CDS 官方 Windows 指南](https://confluence.ecmwf.int/spaces/CKB/pages/121847376/How+to+install+and+use+CDS+API+on+Windows)）。
- 落地：`C:/backup/phd/data/raw/sealevel/satellite-sea-level-global_daily_vdt2024_{year}.zip`
  ——每个 zip 内含该年 365 个日 netCDF（`dt_global_twosat_phy_l4_YYYYMMDD_vDT2024.nc`），
  全量约 **95 GB**（1993–2026）。
- 注意：同一 CDS 账号下**不要并发多个大请求**（实测三路并行互相饿死，
  全部超时零字节；单路顺序反而稳定在 2–10 MB/s）。
- 回退：CMEMS §6 产品；ERA5 风/气压配套见 data_sources.md §6。

## 7. 风 — CCMP v3.1（RSS 开放直链，**免注册**）

- 产品：**RSS CCMP v3.1**（6 小时，0.25°，L4 风场分析；官方页
  <https://www.remss.com/measurements/ccmp/>；记录
  [data.gov](https://catalog.data.gov/dataset/rss-ccmp-6-hourly-10-meter-surface-winds-level-4-version-3-1)；
  [UCAR Climate Data Guide](https://climatedataguide.ucar.edu/climate-data/ccmp-cross-calibrated-multi-platform-wind-vector-analysis)）。
- **开放直链目录**（无需 NASA Earthdata 账号，2026-08 实测）：
  <https://data.remss.com/ccmp/v03.1/Y{YYYY}/M{MM}/>
  - 逐日：`CCMP_Wind_Analysis_YYYYMMDD_V03.1_L4.nc`（约 32 MB/天，
    内含 6 小时步长）；个别日期缺失（如 19930101），以目录清单为准。
  - 每月另有一个 `CCMP_Wind_Analysis_YYYYMM_monthly_mean_V03.1_L4.nc`
    （约 16 MB，当月结束后生成）。
- 下载脚本（清单爬取 + 并发 + 断点续传 + 原子落盘）：
  ```bash
  python scripts/download_ccmp_remss.py                  # 全量镜像（约 390 GB！）
  python scripts/download_ccmp_remss.py --years 2020     # 单年（约 12 GB）
  python scripts/download_ccmp_remss.py --monthly-only   # 只要月均文件（约 6 GB）
  ```
- 落地：`C:/backup/phd/data/raw/ccmp/Y{YYYY}/M{MM}/`（保持远端目录结构，文件名不变；
  `recad ingest` 的通配模板可直接匹配）。
- 规模提醒：全量 ≈ **12,400 日文件 + 400 月文件 ≈ 390 GB**，远超其他源；
  仅做月尺度协变量时优先 `--monthly-only`，或按年份分批拉取。
- 引用要求：使用时注明 RSS CCMP v3.1（Remote Sensing Systems）、NASA
  MEaSUREs 资助；版本与访问日期记入 MANIFEST。

## 8. 水深 — GEBCO 2025（注册后免费）

- <https://www.gebco.net/data_and_products/gridded_bathymetry_data/>
  同意条款后下载 2025 网格（15 弧秒）netCDF → `C:/backup/phd/data/raw/bathymetry/GEBCO_2025.nc`
- 用途：辅助协变量 / 掩膜与地形图（`recad plot` 的 01/06 图可叠加 200 m
  等深线）。

---

## 9. 下载后如何接入管线

```bash
recad download                                   # 先把开放源下到 data/raw
# 手工放置注册源文件到对应 C:/backup/phd/data/raw/XXX/ 目录（见上各节）
recad ingest   --config configs/naccom_1over8.yaml    # 标准化到缓存
recad preprocess --config configs/naccom_1over8.yaml
recad train --config configs/naccom_1over8.yaml       # GPU
```

`recad ingest` 只认 `C:/backup/phd/data/raw/<key>/` 布局，模板在
`configs/*.yaml` 的 `data.templates` 中按文件名填写；文件命名不限
（`{year}`/`{month}` 通配符也支持），详见 `pipeline.ingest_variable`。

## 10. 校验 & 复现记录

- 每个下载文件旁自动生成 `<名>.sha256`；`recad download` 每次运行后刷新
  `data/MANIFEST.md`（来源、状态、文件清单）。
- 重新拉取：删除该源目录后重跑 `recad download --only <name>`；断点续传
  由 HTTP Range 自动处理。
- 注意版权/条款：SOCAT（CC-BY）、NOAA（public domain）、GSHHG、
  CMEMS/GEBCO/NASA 各按其许可使用；发布产品时在论文致谢中标注版本。
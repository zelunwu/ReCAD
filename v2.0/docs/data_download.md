# 数据下载指南（ReCAD v2.0）

所有数据下载到 **`v2.0/data/raw/<source>/`**。`data/` 目录已被
`v2.0/.gitignore` 永久忽略（`data/`），**任何科学数据都不会进入 git**；
可复现性由 `data/raw/MANIFEST.md`（`recad download` 自动生成，记录来源、
URL、文件、校验和）保证。

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
- 落地：`data/raw/xco2air/noaa_gml_mbl/co2_mm_gl.txt`（附 `.sha256`）
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
- 落地：`data/raw/socat/SOCAT_v2026_coast_monthly_lon.._lat.._1993-2021.nc`
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
- 落地：`data/raw/socat_tracks/SOCAT_v2026_decimated_lon.._lat.._1993-2021.nc`
  每行一个观测：`time, latitude, longitude, fCO2_recommended,
  WOCE_CO2_water(QC旗标), dataset_name, sal, temp`。
- 读取/QC/预测因子采样（`recad.data.tracks`）：
  ```python
  from recad.data.tracks import load_tracks, qc_tracks, prepare_point_table
  from recad.data.pipeline import load_prepared
  tr = qc_tracks(load_tracks("data/raw/socat_tracks/*.nc"), qc_flag_max=2)
  tab = prepare_point_table(tr, load_prepared("outputs/prepared.nc"))
  # tab.predictors: [n, 5] = 每个观测处采样的 sst/sss/adt/pco2air/wspd
  ```
- QC 说明：`WOCE_CO2_water <= 2` 视为可用（SOCAT 约定）；fCO2 范围与
  3σ 规则沿用 v1.1（`recad.constants.QC_RANGES`）；**划分训练/验证/测试**
  必须在散点层面按时间或船次进行（点级训练的下一步，见 docs/design.md）。
- 整库（可选）：`files/socat_v2026_fulldata/`、`files/socat_v2026_decimated/`
  目录下有逐船次 CSV，可全量镜像；
  文档数据库 SOCAT v2025：<https://catalog.data.gov/dataset/surface-ocean-co2-atlas-database-version-2025-socatv2025-ncei-accession-0304549>。

## 3. GSHHG 海岸线（开放，约 55 MB）

- 版本：**GSHHG 2.3.7**，Zenodo record [7007502](https://zenodo.org/records/7007502)
  （工具经 Zenodo API 解析最新文件链接）。
- 命令：`recad download --only gshhg`
- 落地：`data/raw/gshhg/gshhg-shp-2.3.7.zip` + 解压目录
- 使用：v2.0 `coastal_mask.method: distance` 可复现海岸掩膜的几何底图
  （`recad ingest` 时把掩膜栅格化成 `data/raw/mask.nc`；或直接下载社区
  预生成掩膜置于 `data/raw/mask.nc`）。

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
- 落地：`data/raw/sst/*.nc`（保持原名即可；`recad ingest` 自动日→月平均并重采样）。
- 注意区分 `access/avhrr/`（AVHRR-only，与 v1.1 相同）与 `access/avhrr-asc/`
  （加 ASCAT，产品页选择时注意）；本仓库默认前者。
- 替代（可选）：MUR 1 km（2002+）、ESA CCI/C3S 0.05°（见 `docs/data_sources.md` §4）。

## 5. SSS — GLORYS12v1（CMEMS，需注册）

- 产品：CMEMS `GLOBAL_MULTIYEAR_PHY_001_030`（GLORYS12V1，1/12°，
  1993→今）。CMEMS 需免费注册：<https://data.marine.copernicus.eu>
- 下载方式（motu-client 或网页子集服务）：
  ```bash
  export CMEMS_USERNAME=you@mail
  export CMEMS_PASSWORD=********
  # 用 CMEMS 官方 motu-client 拉取 surface salinity (so) 月度平均，
  # 存为 data/raw/sss/*.nc（纬度经度需在 0-360 或 -180/180 均可，ingest 会处理）
  ```
- 无 CMEMS 时回退：SOCAT gridded 产品自带的 `coast_salinity_ave_weighted`
  （已在 `socat` 下载中一并取回），或 SMAP V5.0（2015+，沿岸 ~100 km 内
  不可靠，仅作独立校验）。

## 6. ADT/SSH — CMEMS SEALEVEL_GLO_PHY_L4_MY_008_047（需注册）

- 与 §5 同一 CMEMS 账号。产品仍为 0.25° 日平均延迟产品（全球无 1/8°
  L4 网格，见 `docs/data_sources.md` §5）。
- 下载 `adt`（优先，保留平均动力地形）月度平均 → `data/raw/adt/*.nc`。

## 7. 风 — CCMP v3.1（NASA Earthdata，需 token）

- 产品：RSS CCMP v3.1（6 小时，0.25°）。NASA Earthdata 注册：
  <https://urs.earthdata.nasa.gov>
- 下载（cookie/token 认证，NASA 官方文档为准）：
  ```bash
  export EARTHDATA_TOKEN=...
  # 按 data.gov/RSS 索引逐月下载 6 小时风场，聚合月均 u10/v10/wspd
  # 放 data/raw/wspd/*.nc
  ```
- 回退：ERA5（无注册门槛更高，见 data_sources.md §6）。

## 8. 水深 — GEBCO 2025（注册后免费）

- <https://www.gebco.net/data_and_products/gridded_bathymetry_data/>
  同意条款后下载 2025 网格（15 弧秒）netCDF → `data/raw/bathymetry/GEBCO_2025.nc`
- 用途：辅助协变量 / 掩膜与地形图（`recad plot` 的 01/06 图可叠加 200 m
  等深线）。

---

## 9. 下载后如何接入管线

```bash
recad download                                   # 先把开放源下到 data/raw
# 手工放置注册源文件到对应 data/raw/XXX/ 目录（见上各节）
recad ingest   --config configs/naccom_1over8.yaml    # 标准化到缓存
recad preprocess --config configs/naccom_1over8.yaml
recad train --config configs/naccom_1over8.yaml       # GPU
```

`recad ingest` 只认 `data/raw/<key>/` 布局，模板在
`configs/*.yaml` 的 `data.templates` 中按文件名填写；文件命名不限
（`{year}`/`{month}` 通配符也支持），详见 `pipeline.ingest_variable`。

## 10. 校验 & 复现记录

- 每个下载文件旁自动生成 `<名>.sha256`；`recad download` 每次运行后刷新
  `data/MANIFEST.md`（来源、状态、文件清单）。
- 重新拉取：删除该源目录后重跑 `recad download --only <name>`；断点续传
  由 HTTP Range 自动处理。
- 注意版权/条款：SOCAT（CC-BY）、NOAA（public domain）、GSHHG、
  CMEMS/GEBCO/NASA 各按其许可使用；发布产品时在论文致谢中标注版本。
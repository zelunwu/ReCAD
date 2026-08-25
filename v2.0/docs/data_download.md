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
  recad download --only socat --region -100,-40,10,65 --years 1993,2021   # NACCOM 窗口
  # 全球：直接把窗口换成全局 --region -180,-180,-90,90？不对——
  # SOCAT 经度为 0-360，见下：
  recad download --only socat --region 0,360,-78,84 --years 1993,2021      # 全球窗口（大！）
  ```
- 落地：`data/raw/socat/SOCAT_v2026_coast_monthly_lon.._lat.._1993-2021.nc`
- 手工方式（浏览器）：打开
  <https://data.pmel.noaa.gov/socat/erddap/griddap/SOCAT_v2026_qrtrdeg_gridded_coast_monthly.html>
  选择变量与窗口导出 `.nc`。

## 3. GSHHG 海岸线（开放，约 55 MB）

- 版本：**GSHHG 2.3.7**，Zenodo record [7007502](https://zenodo.org/records/7007502)
  （工具经 Zenodo API 解析最新文件链接）。
- 命令：`recad download --only gshhg`
- 落地：`data/raw/gshhg/gshhg-shp-2.3.7.zip` + 解压目录
- 使用：v2.0 `coastal_mask.method: distance` 可复现海岸掩膜的几何底图
  （`recad ingest` 时把掩膜栅格化成 `data/raw/mask.nc`；或直接下载社区
  预生成掩膜置于 `data/raw/mask.nc`）。

## 4. SST — OISST v2.1（0.25°，日平均；需自行批量下载）

- 版本：**OISST v2.1**（NCEI，仍是当前主产品；无 v3）。
- 入口（NCEI 归档/云分发，HTTPS 模式随分发方不同，勿硬编码）：
  - 产品页：<https://www.ncei.noaa.gov/products/optimum-interpolation-sst>
  - THREDDS：<https://www.ncei.noaa.gov/thredds-ocean/catalog/oisst-base/catalog.html>
- 最小方案（先验证流程）：
  ```bash
  # 直接在浏览器/NCEI 归档选取 1993 年 1 月的 daily 文件（~15 MB/天），
  # 或下载月度聚合文件 OISST-V2.1-AVHRR_19930101-...v02r01 系列。
  # 放入：
  mkdir -p data/raw/sst
  # 每个文件命名为 oisst 任意 *.nc 均可；recad ingest 会日→月平均并重采样到目标网格
  ```
- 需求规模：全球月度化 SST 的最小集 = 每期 monthly aggregate（~10 GB/年
  daily原始估算 → 建议只取 monthly mean 产品）。
- 替代（可选）：MUR 1 km（2002+）、ESA CCI/C3S 0.05°（见
  `docs/data_sources.md` §4）。

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
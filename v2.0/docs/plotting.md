# 绘图函数总结与移植（v1/v1.1 → v2.0 `recad.viz`）

本文档把 v1.1（及 v1 论文版）里所有散落在 notebook/live-script 里的绘图代码
逐条总结，并说明它们在 v2.0 中的工具化位置。**v2.0 将绘图从 notebook 代码
提升为 `src/recad/viz/` 下的可复用、可测试工具**，并新增 `recad plot` 命令
一键重绘整套图。

## 1. 原版绘图函数清单（来源）

v1.1 的绘图全部散落在四个文件中：

| 文件 | 内容 |
|---|---|
| `v1.1/Figures_Maintext.ipynb` | 主文图：地形、季节/月气候态地图、train/val/test 散点密度、区域散点、统计表、气候态对比、趋势图/趋势序列、不确定度分量图 |
| `v1.1/Figure_Uncertainties.ipynb` | 不确定度分量地图（Measured/Grid/Model/Inputs/All 2×3）、区域 RMSE 与不确定度表 |
| `v1.1/u_inputs_Monte_Carlo.mlx` | 输入误差 MC 各分量 std 的 2×3 直方图 |
| `v1/Figures_Dissertation_Chapter3.ipynb` | 论文图（Figure_01/Topo、Model Training/Validation/Test、Seasonality、Trend、Bias、Statistics、Comparison、S01/S02） |

## 2. 核心辅助函数（原样总结）

1. **`bin_counts(x, y, xlim, ylim, step)`** —— 2-D 直方图密度分箱，
   `dens[y_bin, x_bin]` + 边界数组；所有“预测 vs 观测”散点密度图（Figure_03/Model_Training、
   S01/S02）都以它打底。→ `recad.viz.scatter.bin_counts`（逐行移植）。
2. **`calc_trend_Sutton(t, y, w)`** —— Sutton et al. (2007) 两步趋势法：
   (a) HAC-OLS 粗去趋势；(b) 残差月气候态 − 年气候态 = 季节调整项；
   (c) 去季节序列上做 WLS 得最终趋势（HAC 协方差，maxlags=1）。
   返回 `monthly_clim / annual_clim / ts_monthly_desaison / model`。
   → `recad.viz.trends.calc_trend_sutton`（statsmodels 存在时逐行复刻；
   无 statsmodels 时用 NumPy OLS 回退，文档注明斜率一致、标准误略有差异）。
3. **`calculate_metrics(y_true, y_pred)`** —— 图内统计框的 R²(1−SS_res/SS_tot)
   与 RMSE（注意与 `calculateR2RMSE.m` 的 corr² 定义不同，二者用途不同）。
   → `recad.viz.scatter.calculate_metrics`。
4. **`calc_clim_anom / get_clim_anom`** —— 月气候态与距平分解。
   → `recad.data.ingest.calc_clim_anom`（数据管线已在 v2.0 中承载）。
5. **区域掩膜**（GoMeSS/GoMe/SS/GStL/SAB/MAB/GoMx/CbS/Atlantic North/South；
   分析框 LAS/WFS/MAB/North/S.GStL）—— 手写 lon/lat 几何（含 GoMx 两条斜线裁切）。
   → `recad.viz.regions.region_masks`（纯几何、可测试、可复现）。

## 3. 绘图类型 → v2.0 工具映射

| v1.1 图形 | 画法要点 | v2.0 工具 |
|---|---|---|
| 单幅地图（mean/product/bias/trend） | cartopy PlateCarree + `pcolor` + coastlines/LAND/RIVERS/网格/格式化刻度 | `recad.viz.geo.field_map` + `setup_geo_axes` |
| 多面板共享色标（季节 2×2、月气候态 3×4、不确定度分量 2×3） | `AxesGrid` + 单一 colorbar | `recad.viz.geo.panel_maps` |
| 区域分界线与标签（GStL&GB/SS/GoMe/MAB/SAB/GoMx） | 折线 + 文本 | `recad.viz.geo.add_region_boundaries` |
| 地形图（ETOPO2 + 200 m contour + bathymetry colormap） | 自定义 colormap/levels | `recad.viz.geo.field_map` + 用户 colormap（数据层提供 field） |
| 预测 vs 观测密度散点 + 1:1 线 + 统计框 | `bin_counts` + pcolor + OLS R²/RMSE/MAE/MBE/N 文本框 | `recad.viz.scatter.density_scatter`（`bin_counts`/`metrics_text`/`calculate_metrics`） |
| 月气候态 errorbar 对比（SOCAT vs 产品，差异文本框） | `errorbar(mean ± std)` + 差值注释 | `recad.viz.series.climatology_comparison` / `monthly_climatology` |
| 去季节时间序列 + Sutton 趋势线 + trend±err/p 注释 | 散点 + 拟合线 + 统计文本 | `recad.viz.series.deseasonalized_timeseries` + `trends.calc_trend_sutton` |
| 逐格线性趋势地图（µatm yr⁻¹） | 每格 OLS | `recad.viz.trends.trend_map` |
| 不确定度分量地图（u_SSS/u_SST/u_SSH/u_pCO2air/u_inputs） | 共享色标面板 | `recad.viz.uncertainty.uncertainty_panels` |
| MC 各分量 std 直方图 2×3 | `hist` | `recad.viz.uncertainty.contribution_histograms` |
| 统计表（Region×Type 的 R²/RMSE/MAE/MBE；不确定度表） | pandas → xlsx | `recad.viz.report.model_summary_table` / `uncertainty_table` → CSV |
| **整套图集一键重绘** | – | `recad.viz.report.render_figure_set` + CLI `recad plot` |

## 4. `recad plot` 用法

```bash
recad plot --product outputs/ReCAD-v2.0-pCO2.nc \
           --prepared outputs/prepared.nc --masks outputs/masks.nc \
           --outdir figures --fmt png --fmt pdf
```

输出（对应 v1.1 图的 `01_…`~`09_…` 命名）与两张统计表（`model_summary.csv`、
`uncertainty_table.csv`）；对所有 NA 大西洋域网格自动套用 v1.1 六大子区域，
其他区域则回退为全域。

## 5. 依赖与降级策略

- `cartopy`、`statsmodels`、`openpyxl` 为可选（`pip install -e ".[viz]"`）。
- 无 cartopy：地图退化为普通 axes（无海岸线/陆地），同一套代码在 CI 可跑。
- 无 statsmodels：`calc_trend_sutton` 用 NumPy OLS（斜率一致；标准误与
  HAC 版本略有差异，结果对象用 `using_statsmodels=False` 标注）。
- 字体/图面统一走 `recad.viz.style.use_v11_style()`（论文风格 rcParams）。

## 6. 已修正的原版问题（诚实记录）

1. v1.1 面板代码把 `plt.show()`、硬编码路径和大量注释混杂在一起；
   v2.0 全部参数化并测试（`tests/test_viz.py`）。
2. v1.1 图内统计框使用 `statsmodels.tools.eval_measures.rmse(x, y)`
   （x=pred, y=obs，顺序一致）；v2.0 的 `metrics_text` 保持同样顺序与含义。
3. v1.1 区域掩膜硬编码于 notebook，v2.0 提为 `regions.region_masks`，
   并用单元测试锁定 GoMx 斜线裁剪与 Atlantic 并集语义。
4. 趋势图中 v1.1 把“月气候态”与“年气候态”混在同一字典里使用；
   v2.0 `TrendResult` 明确区分 `monthly_clim` 与 `annual_clim`。
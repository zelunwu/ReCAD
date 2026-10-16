# Reviewer-ready 实验归档

本目录保存可直接用于论文复核和 reviewer 回复的版本化证据。每个支撑科学结论的正式实验必须建立 `docs/experiment_archive/<experiment_id>/`，包含完整报告、图片、逐图 caption、图片源 CSV、复现入口和哈希清单。每张图片必须嵌入 `REPORT.md`，完整 caption 紧跟在对应图片下方；`CAPTIONS.md` 只作为集中索引。归档必须通过：

```powershell
python scripts/verify_experiment_archive.py docs/experiment_archive/<experiment_id>
```

大型原始数据、逐行预测和 checkpoint 留在被 Git 忽略的 `outputs/`，由归档 manifest 记录路径和 SHA256。归档一经用于结论就不静默改写；实质变化使用新的 experiment ID 或版本。

## 已归档实验

| 实验 | 状态 | 报告 |
|---|---|---|
| `p1_sss_viability_v2.2`（Issue #7） | verified；development evidence | [完整报告](p1_sss_viability_v2.2/REPORT.md) |
| `p1_fco2_viability_v2.2`（Issue #8） | verified；diagnostic_only | [完整报告](p1_fco2_viability_v2.2/REPORT.md) |
| `p1_ta_viability_v2.2`（Issue #9） | verified；diagnostic_only | [完整报告](p1_ta_viability_v2.2/REPORT.md) |
| `p1_ta_bights_v2.2`（Issue #15） | verified；SAB fail；MAB diagnostic_only | [完整报告](p1_ta_bights_v2.2/REPORT.md) |
| `p1_ta_sab_latitude_sensitivity_v2.2`（Issue #17） | verified；27°N 假设未支持；全部方案未通过完整 gate | [完整报告](p1_ta_sab_latitude_sensitivity_v2.2/REPORT.md) |
| `p1_ta_sss_sab_decadal_trends_v2.2`（Issue #19） | diagnostic；TA not identifiable；SSS diagnostic_only | [完整报告](p1_ta_sss_sab_decadal_trends_v2.2/REPORT.md) |
| `p1_ta_sss_mab_decadal_trends_v2.2`（Issue #21） | diagnostic；TA not identifiable；SSS diagnostic_only | [完整报告](p1_ta_sss_mab_decadal_trends_v2.2/REPORT.md) |
| `p1_dic_inverse_co2sys_v2.2`（Issue #10） | diagnostic_only；exact solver pass；upstream/calibration gates fail | [完整报告](p1_dic_inverse_co2sys_v2.2/REPORT.md) |
| `p2_scope_reconciliation_v2.3`（Issue #36） | verified；scope_reconciled；no new labels opened | [完整报告](p2_scope_reconciliation_v2.3/REPORT.md) |
| `p2_benchmark_freeze_v2.3`（Issue #37） | verified；benchmark_contract_frozen；no new labels opened | [完整报告](p2_benchmark_freeze_v2.3/REPORT.md) |
| `p2_global_coastal_cache_v2.3`（Issue #38） | verified；pass_data_freeze；global observation cache；legacy product-mask defect exposed | [完整报告](p2_global_coastal_cache_v2.3/REPORT.md) |
| `p2_global_coastal_mask_v2.3`（Issue #50） | verified；pass_geometry_freeze；400 km global geometry；GEBCO 200 m shelf core | [完整报告](p2_global_coastal_mask_v2.3/REPORT.md) |
| `p2_predictor_availability_v2.3`（Issue #52） | verified；pass_availability_audit；three dense dynamic fields；masked GLORYS SSS；optional ADT | [完整报告](p2_predictor_availability_v2.3/REPORT.md) |
| `p2_global_predictor_cube_v2.3`（Issue #53） | verified；pass_cube_freeze；396 months × 838,661 nodes；zero-mismatch replay | [完整报告](p2_global_predictor_cube_v2.3/REPORT.md) |
| `p2_model_zoo_v2.4`（Issue #62） | verified；development_model_zoo_complete；no production promotion；graph diagnostic triggered | [完整报告](p2_model_zoo_v2.4/REPORT.md) |
| `p2_model_family_benchmark_v2.5` (Issue #88) | verified; preregistered benchmark contract; no models ranked | [report](p2_model_family_benchmark_v2.5/REPORT.md) |
| `p2_process_predictors_v2.4`（Issue #63） | verified；fCO2 continuous background retained for development only；all process proxies negative；no production promotion | [完整报告](p2_process_predictors_v2.4/REPORT.md) |
| `p2_conditional_combination_v2.4`（Issue #64） | verified；fCO2 convex ensemble retained for development only；graph/scale-head negative | [完整报告](p2_conditional_combination_v2.4/REPORT.md) |
| `p2_global_reliability_contract_v2.4`（Issue #44） | verified；complete A/B/C/D product projection；global shifted-interval qualification failed | [完整报告](p2_global_reliability_contract_v2.4/REPORT.md) |
| `p2_na_carbonate_optional_v2.4`（Issue #45） | trigger audit；untriggered/non-blocking；no labels opened | [完整报告](p2_na_carbonate_optional_v2.4/REPORT.md) |
| `p3_independent_freeze_v2.4`（Issue #5 preparation） | candidate/protocol freeze；waiting for eligible post-v2026 independent source；no labels opened | [完整报告](p3_independent_freeze_v2.4/REPORT.md) |

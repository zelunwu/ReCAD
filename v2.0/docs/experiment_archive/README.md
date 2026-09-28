# Reviewer-ready 实验归档

本目录保存可直接用于论文复核和 reviewer 回复的版本化证据。每个支撑科学结论的正式实验必须建立 `docs/experiment_archive/<experiment_id>/`，包含完整报告、图片、逐图 caption、图片源 CSV、复现入口和哈希清单，并通过：

```powershell
python scripts/verify_experiment_archive.py docs/experiment_archive/<experiment_id>
```

大型原始数据、逐行预测和 checkpoint 留在被 Git 忽略的 `outputs/`，由归档 manifest 记录路径和 SHA256。归档一经用于结论就不静默改写；实质变化使用新的 experiment ID 或版本。

## 已归档实验

| 实验 | 状态 | 报告 |
|---|---|---|
| `p1_sss_viability_v2.2`（Issue #7） | verified；development evidence | [完整报告](p1_sss_viability_v2.2/REPORT.md) |

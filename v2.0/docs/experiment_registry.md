# ReCAD v2 实验总账

本文件是实验记录的统一入口。它回答四个问题：做过什么、使用了什么数据与 split、如何选择模型、结论能信到什么程度。详细协议、机器可读 manifest、指标和大文件仍保存在各实验目录；本文件只保存可进入 Git 的元数据和索引。

## 记录层级

1. `docs/experiment_protocol.md`：全项目共同的验证原则和统计口径。
2. `configs/experiment_record_template.yaml`：每个新正式实验在运行前填写的预登记记录。
3. `outputs/experiments/<experiment_id>/`：本地机器可读产物，包括 `protocol.json`、`data_manifest.json`、`selection.json`、metrics、预测和 checkpoint；该目录不进入 Git。
4. 本文件：跨实验总账、证据等级、已打开测试集和失效实验。
5. GitHub Roadmap Issue：未来任务、依赖关系和通过门槛，不代替实验记录。

## 证据等级

| 等级 | 含义 |
|---|---|
| A | 运行前协议和数据 manifest 完整；grouped CV/开发选择清楚；测试状态可审计；多种子或预登记复现完成 |
| B | 方法和主要结果可复现，但 manifest、预登记、外层验证或多种子中至少一项不完整 |
| C | 探索、诊断或工程 smoke；只能产生下一步假设，不能支撑论文主结论 |
| X | 已确认输入、路径、标签泄漏或实现错误；产物保留用于审计，不参与模型比较 |

`2004–2005` 已被多轮查看，所有相关结果只能标为历史 benchmark，不能再作为独立验证。`independent` 旧字段名不自动代表统计独立。

## v2.2 正式实验入口

P0 当前以 `configs/frozen/data_manifest_v2.2.json` 为准：数据轴保留至 2026，SSS/fCO2 核心观测期到 2025，TA/DIC 观测期到 2024，TA/DIC 模型输出可延伸至 2026。验收报告见 `docs/p0_freeze_report_v2.2.md`。v2.1 因错误排除完整 2025 数据而失效。首轮 P1 实验为：

| 实验 ID | 目标 | 状态 | checkpoint 选择 |
|---|---|---|---|
| `p1_sss_baselines_v2.2` | SSS 背景订正与软空间专家 | preregistered | development LME macro-RMSE |
| `p1_fco2_baselines_v2.2` | fCO2 基线与软空间专家 | preregistered | development LME macro-RMSE |
| `p1_ta_baselines_v2.2` | 北美 TA 局地线性/层级部分池化，DIC 结构派生 | preregistered | grouped development LME macro-RMSE |

41 个 CODAP 2022–2024、未匹配 GLODAP 的完整航次已标记 `external_independent`。未来 SOCAT 相对 v2026 的新增航次是 SSS/fCO2 外部集；P3 前不得查看标签。

## 历史实验索引

| 实验 ID | 主要问题 | 数据/协议记录 | 主要结果 | 等级与用途 |
|---|---|---|---|---|
| `point_mlp` | 单点 MLP 基线 | 结果目录只有汇总指标 | `metrics_summary.csv` | C；骨干初筛 |
| `unet` | U-Net 空间基线 | 结果目录只有汇总指标 | `metrics_summary.csv` | C；骨干初筛 |
| `st_small` | 小型 ST 初始基线 | 旧 split；无独立 manifest | `metrics_summary.csv` | C；历史基线 |
| `st_full` | 大型 fCO2-only ST | 方法见 `docs/experiment_protocol.md` 尾部 | 时间测试 24.48 µatm，空间挑战 39.38 µatm | B；空间挑战不是外部验证 |
| `st_audit_20260906` | 训练收敛、标签曝光和 checkpoint 审计 | `audit.json` | 发现 dev 震荡、旧早停覆盖不完整 | C；诊断证据 |
| `controlled_20260906` | 容量、优化、局地解码受控比较 | `protocol.json`、`data_manifest.json`、源码哈希；报告 `docs/backup/controlled_experiments_20260906.md` | 最佳 ST96 MSE LR1e-3，dev RMSE 27.326 | A（开发实验）；未做新独立验证 |
| `chla_ablation_20260906` | MODIS Chl-a 是否改善 fCO2 | protocol/data/selection 完整；报告 `docs/backup/chla_ablation_20260906.md` | 共同覆盖期 dev RMSE 中位数改善 6.28% | A（开发消融）；2004–2005 仅 benchmark |
| `unified_chla_full_20260906_r2` | Chl-a 统一缺失处理与完整复现 | protocol/data/selection 完整 | `metrics_by_seed.csv`、`summary.csv` | A（开发实验） |
| `joint_four_full_20260907` | 四目标独立头与软 CO2SYS 约束 | protocol/data、3 seeds；报告在实验目录 | 闭合大幅改善，预测技能仅小幅变化 | B；旧 test 已打开 |
| `joint_regularized_20260908` | 稀疏 TA/DIC 正则与采样 | `protocol.json` | `cv_results.json`、`metrics.csv` | B；阶段性候选 |
| `joint_adapter_direct_20260908` | 冻结/adapter 碳参数适配 | `protocol.json` | 未超过基线 | B；负结果保留 |
| `joint_regularization_comparison_20260908` | 正则方案横向比较 | 报告完整，manifest 不完整 | `all_metrics.csv`、`cv_summary.csv` | B |
| `joint_r0_r4_20260908` | R0–R4 五折受控比较 | protocol/selection；报告在实验目录 | R0 CV 最佳，容量不是主要瓶颈 | A（内部 grouped CV）；无外部盲测 |
| `carbonate_parameterization_20260909` | CO2SYS 参数化灵敏度 | 报告 `docs/backup/carbonate_identifiability_redesign_20260909.md` | 支持 `fCO2+TA→DIC` | B；结构设计证据 |
| `carbonate_latent_20260909` | 东岸结构化 TA 与派生 DIC | protocol/selection、五折与多种子 | 东岸仍保留旧 B2；TA 区间偏窄 | B |
| `us_west_carbonate_latent_20260909` | 西岸结构化碳酸盐模型 | protocol/selection、五折与多种子 | 留航次 TA RMSE 约 75–127 | B；未达到生产门槛 |
| `us_west_*_invalid_sal_path` | 错误复用了东岸 SAL 路径 | 目录名显式标记 invalid | 不使用 | X |
| `weighted_naccom_v1/v2/v3` | 支持度加权化学约束 | 指标存在，protocol/manifest 不完整 | 见 `docs/backup/weighted_carbonate_results.md` | B/C；只作路线诊断 |
| `weighted_global_v1` | 全球加权联合模型初版 | 指标存在，protocol/manifest 不完整 | 后发现 SSS 目标泄漏 | X |
| `weighted_global_v2` | 修复泄漏后的全球基线 | 结果见 `docs/current_status_and_roadmap_20260928.md` | SSS 可用；fCO2 validation RMSE 60.45、skill -0.138 | B；当前全球有效基线 |
| `smoke_sel` | checkpoint selection 工程检查 | 无正式 protocol | 因 `selection_score` 未定义失败 | C；修复前不得正式重训 |

缓存构建目录（`global_joint_cache*`、`us_west_joint_cache*`、`us_west_unified_cache*`）是数据工程产物，不作为模型实验计数；其 `data_manifest.json` 仍是上表相关实验的数据谱系组成部分。

### 工程运行与汇总目录

这些目录也由自动审计扫描，但不产生独立科学结论：

| 目录 | 类型 | 处置 |
|---|---|---|
| `gsmoke` | 全球管线 smoke | C；只证明代码路径曾运行 |
| `weighted_smoke` | 加权训练 smoke | C；只作工程检查 |
| `ws2` | weighted runner 临时/续跑目录 | C；指标不单独用于论文 |
| `unified_chla_full_20260906` | Chl-a 首次完整运行目录 | C；由修正并有 protocol 的 `_r2` 替代 |
| `us_coasts_carbonate_report_20260909` | 东西岸结果汇总 | 不是新训练；其结论归属于 east/west carbonate latent 实验 |
| `global_joint_cache`、`global_joint_cache_p32` | 数据缓存构建 | 非模型实验；manifest 作为输入谱系 |
| `us_west_joint_cache_20260909`、`us_west_unified_cache_20260909` | 西岸缓存构建 | 非模型实验；只使用无 `_invalid_sal_path` 的版本 |
| `us_west_joint_cache_20260909_invalid_sal_path` | 错误缓存 | X；保留审计，禁止使用 |

`scripts/audit_experiment_records.py` 会列出 `outputs/experiments/` 中每一个目录及其 protocol/manifest/selection/report/metrics/checkpoint 完整性。该脚本只检查文件存在性，不能根据文件名判定统计独立性或证据等级。

## 可横向比较的关键数值

### fCO2 早期骨干比较（旧 NACCOM split）

| 模型 | train RMSE | dev RMSE | 2004–2005 benchmark RMSE | 空间挑战 RMSE | 解释 |
|---|---:|---:|---:|---:|---|
| Point MLP | 27.09 | 29.45 | 24.26 | 40.38 | 单点基线 |
| U-Net | 26.67 | 28.65 | 26.50 | 39.56 | 空间卷积基线 |
| ST-small | 28.46 | 29.67 | 25.69 | 38.68 | 小型时空模型 |
| ST-full | 26.46 | 27.95 | 24.48 | 39.38 | dev 改善但空间挑战未超过 ST-small |

这些数值使用旧 split，只用于历史骨干比较。`controlled_20260906` 在未打开新 test 的开发实验中将 ST96 MSE LR1e-3 选为最佳：selected step 2750，train/dev RMSE 20.736/27.326，dev R² 0.653。

### Chl-a 消融（共同覆盖期）

| 条件 | 三种子 dev RMSE 中位数 | 历史 benchmark RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| baseline | 25.743 | 23.115 | 14.796 | 0.618 |
| with Chl-a | 24.127 | 21.099 | 13.456 | 0.682 |

有效结论限于 MODIS 共同覆盖期和开发消融；benchmark 已打开。

### 四目标与稀疏碳参数实验

| 实验/条件 | 关键结果 | 当前解释 |
|---|---|---|
| B1 no-chem | dev SSS/fCO2/DIC/TA RMSE = 0.917/26.456/46.255/59.550 | 无化学项基线 |
| B2 soft-CO2SYS | dev = 0.927/27.293/43.413/48.984 | TA/DIC 改善，fCO2 略退化；闭合改善不能单独证明泛化 |
| R0 protocol | internal test = 0.860/25.409/40.983/47.298 | R0 为 R0–R4 五折最佳，CV score 0.4529±0.0201 |
| R2 less-GLODAP | internal test = 0.859/26.338/40.134/50.689 | 降低碳样本频率没有解决泛化 |
| East C2 probabilistic | CV score 0.3807±0.0101 | 东岸结构候选，但未超过旧 B2 的全部门槛 |
| West C1 structural | CV score 0.5483±0.0345 | 西岸最佳结构候选，航次外 TA 仍不可接受 |

四元组顺序均为 SSS PSU / fCO2 µatm / DIC µmol kg-1 / TA µmol kg-1。不同实验的数据和 split 不完全相同，不可只按表中绝对值排名。

### 当前全球有效基线

`weighted_global_v2` 修复了 SSS 目标泄漏：

| 区域/来源 | split | 目标 | N | RMSE | R² |
|---|---|---|---:|---:|---:|
| global SOCAT | validation | SSS | 220,295 | 1.156 | 0.987 |
| global SOCAT | validation | fCO2 | 239,396 | 60.450 | 0.210 |
| global SOCAT | old benchmark | SSS | 40,844 | 1.081 | 0.986 |
| global SOCAT | old benchmark | fCO2 | 48,934 | 53.198 | -0.294 |
| global GLODAP | validation | TA | 2,824 | 41.764 | 0.791 |
| global GLODAP | validation | DIC | 2,824 | 42.071 | 0.722 |

全球 SSS 的高 R²受极端盐度扩大方差影响；美国窗口 skill 仅约 0.39–0.46。全球 fCO2 尚未通过产品门槛。

## 记录完整性缺口

| 类别 | 缺口 | 处理 |
|---|---|---|
| `point_mlp/unet/st_small/st_full` | 无统一 data manifest 和预登记 | 保留为 C/B 历史骨干，不补写成预登记 |
| weighted 系列 | 多数缺 protocol/manifest/selection | 保留已有指标；新路线使用新 ID 和模板 |
| carbonate latent | 有 protocol/selection，缺统一数据 manifest | 证据为 B；CODAP 新数据重训必须建立 v2.1 manifest |
| cache 构建 | manifest 分散，部分错误缓存仍在 | 错误目录保留 `_invalid_*`，总账标 X |
| 外部验证 | TA/DIC 已冻结 41 个新 CODAP 航次；SSS/fCO2 为未来 SOCAT 增量，标签尚不可用 | 开发期只使用 grouped CV/development；P3 一次性开启外部标签 |

## 新实验必须记录的字段

运行正式训练前，从 `configs/experiment_record_template.yaml` 复制一份到版本控制目录，冻结以下内容：

- 科学问题、假设、唯一主比较和停止规则；
- 数据产品版本、输入文件哈希、QC、时间范围、空间范围和 target provenance；
- split manifest 哈希、group 定义、embargo、locked test 和 external 数据状态；
- 输入、输出、背景场、目标变换和缺失处理；
- 模型、损失、采样、参数量、种子、optimizer steps、GPU 小时预算；
- checkpoint 选择规则、主指标、区域宏平均、worst-group 和不确定性覆盖；
- Git commit、工作树是否干净、配置和源码哈希；
- 结果目录、checkpoint、预测、metrics、图和报告路径；
- 失败、偏离预登记、测试开启时间和结论可信范围。

实验完成后更新本总账。探索运行不得事后补写成“预登记”；必须标为 C。任何输入错误保留原目录并降为 X，修复后使用新的 experiment ID。

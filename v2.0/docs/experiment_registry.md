# ReCAD v2 实验总账

本文件是实验记录的统一入口。它回答四个问题：做过什么、使用了什么数据与 split、如何选择模型、结论能信到什么程度。详细协议、机器可读 manifest、指标和大文件仍保存在各实验目录；本文件只保存可进入 Git 的元数据和索引。

> **2026-10-13 证据更正**：#40–#42 仍保留其点预测开发比较，但不再视为 untouched、严格新地理位置或可部署不确定性证据。元数据审计确认旧 SOCAT locked/external 组进入 P2 manifest 与 matchup；现行 `spatial_fold` 是 dominant-block 航次分组而非实际 geographic block 留出；#40 SSS 区间按观测真值盐度分箱；#40–#42 校准没有完全嵌套在当前 outer-training 中。受影响的 sealed、strict-spatial、leakage-safe calibration、coverage 和 risk-coverage 表述均由 #61 重算并更正。已暴露标签永久降为 development，不得通过重新分割恢复 blind 身份。

## 记录层级

1. `docs/experiment_protocol.md`：全项目共同的验证原则和统计口径。
2. `configs/experiment_record_template.yaml`：每个新正式实验在运行前填写的预登记记录。
3. `outputs/experiments/<experiment_id>/`：本地机器可读产物，包括 `protocol.json`、`data_manifest.json`、`selection.json`、metrics、预测和 checkpoint；该目录不进入 Git。
4. `docs/experiment_archive/<experiment_id>/`：进入 Git 的 reviewer-ready 报告、图片、图片源表和哈希清单；正式实验完成与关闭 Issue 的强制门槛。
5. 本文件：跨实验总账、证据等级、已打开测试集和失效实验。
6. GitHub Roadmap Issue：未来任务、依赖关系和通过门槛，不代替实验记录。

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
| `p1_sss_baselines_v2.2` | SSS 背景订正与软空间专家 | development complete；reviewer archive verified；进入 #11 | soft experts by development LME macro-RMSE |
| `p1_fco2_baselines_v2.2` | fCO2 季节趋势气候态与残差模型 | development complete；archive verified；`diagnostic_only` | CatBoost by development LME macro-RMSE；gate failed on worst-LME and >100 km support |
| `p1_ta_baselines_v2.2` | 北美 TA 局地线性/层级部分池化，DIC 结构派生 | development complete；archive verified；`diagnostic_only` | hierarchical residual ensemble；gate failed on cruise-equal RMSE and distant-support bins |
| `p1_ta_bights_v2.2` | SAB 与 MAB 分区独立 TA 可行性 | development complete；archive verified；SAB `fail`，MAB `diagnostic_only` | SAB hierarchical TA-SSS；MAB hierarchical residual ensemble；locked/external sealed |
| `p1_ta_sab_latitude_sensitivity_v2.2` | SAB TA 南边界敏感性（26/27/28.45/30.5°N） | development/grouped-CV complete；archive verified；无方案通过完整 gate | 不支持“27°N 以南导致退化”；28.45–30.5°N 对比受模型族切换混杂，只生成后续假设 |
| `p1_ta_sss_sab_decadal_trends_v2.2` | SAB 表层 TA/SSS 观测能否识别年代际趋势 | diagnostic complete；archive verified；locked/external sealed | TA 因航次稀少和 2014 集中而不可识别；SSS 为弱负趋势但对截止年份/区域敏感，只能 diagnostic_only |
| `p1_ta_sss_mab_decadal_trends_v2.2` | MAB 表层 TA/SSS 观测能否识别年代际趋势 | diagnostic complete；archive verified；locked/external sealed | TA 原始负趋势经空间季节校正后消失；SSS 全区与纬带方向冲突，只能 diagnostic_only |
| `p1_applicability_v2.2` | SSS/fCO2/TA/DIC 无泄漏外层残差与定量适用域 | complete；2,230,337 条 nested outer predictions；7,361,632 条 2025–2026 grid-month support records；archive verified；locked/external sealed | SSS/fCO2 保留 environmental-k64，选择性风险 AUC 相对最佳简单基线改善 9.6%/11.2%；TA/DIC 不满足单调误差要求，只保留区域级 diagnostic support |
| `p1_sss_reliability_v2.2` | SSS 支持度收缩、校准区间、A/B/C/D atlas 与一次性 locked audit | complete；75,627 条 locked；1,840,408 条 2025–2026 grid-month；archive verified；external sealed；`diagnostic_only` | locked RMSE 1.071 PSU vs GLORYS 1.804，skill 0.647；B 级 MAE 0.446；50/90 coverage 0.484/0.884；North Brazil Shelf 最差 LME 比值 1.355 超过冻结 1.10 门槛，故不得升级为正式产品 |

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

## P1 共享运行基础设施

`p1_framework_v2.2`（GitHub Issue #6）冻结了 P1 的唯一数据访问和评估接口。实现位于 `src/recad/evaluate/p1_framework.py`，配置为 `configs/p1_framework_v2.2.yaml`，接口和运行方式见 `docs/p1_shared_evaluation_framework.md`。真实 v2.2 数据的机器审计写入被 Git 忽略的 `outputs/audits/p1_framework_v2.2.json`；审计只读取 train/development 标签，未开启 locked test 或 external-independent 标签。

真实数据 smoke audit 的可用标签为：SSS train/development 332,295/72,161，fCO2 359,603/78,669，TA 4,515/925，DIC 4,516/840。SSS 和 fCO2 的开发标签覆盖至 2025；2026 严格输入可用位置共 140,746 个节点月，全部只能输出为 provisional。

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
- reviewer-ready 归档路径、完整报告、作图源表、manifest 和自动校验状态；
- 失败、偏离预登记、测试开启时间和结论可信范围。

实验完成后更新本总账。正式实验必须先通过 `python scripts/verify_experiment_archive.py docs/experiment_archive/<experiment_id>` 才能标记完成或关闭 Issue。探索运行不得事后补写成“预登记”；必须标为 C。任何输入错误保留原目录并降为 X，修复后使用新的 experiment ID。

## P1.5c fCO2 支持感知区域产品（Issue #26）

`p1_fco2_reliability_v2.2` 比较了 CatBoost、标准化 MSE MLP、30 µatm Huber MLP、严格 OOF-SSS CatBoost 和四种支持收缩。全局阶段选择 OOF-SSS CatBoost + linear 2–5，但整 LME 留出没有 A/B 区域，因此不支持全球声明。预登记的条件区域专家分支用 cruise OOF 提名 LME 6/12，再由空间块和前向时间确认；最终只有 Caribbean Sea（LME 12）通过：

| 外层验证 | N | RMSE (µatm) | MAE (µatm) | q90 absolute error | skill vs background | 50/90 coverage |
|---|---:|---:|---:|---:|---:|---:|
| cruise | 78,059 | 9.538 | 6.140 | 12.561 | 0.364 | 0.500 / 0.900 |
| spatial block | 78,059 | 10.033 | 6.249 | 12.894 | 0.295 | 0.498 / 0.894 |
| forward 2019–2021 | 15,252 | 13.571 | 9.038 | 20.094 | 0.161 | 0.523 / 0.850 |

状态为 **B / `pass_regional`**，只允许提名未来一次性 locked fCO2 audit；locked 与外部独立标签均未开启。2025 core + available-2026 provisional atlas 含 85,501 个 A 和 1,754,907 个 D grid-month；非 LME 12 或支持不足位置均为背景回退。Chl-a 在完全相同历史支持集上使四种外层 RMSE 改善 1.75–5.53%，但缓存止于 2020-12，未进入 2025 产品。完整报告、8 张图及逐图 caption、9 个源表见 `docs/experiment_archive/p1_fco2_reliability_v2.2/`；大文件位于本地忽略目录 `outputs/experiments/p1_fco2_reliability_v2.2/`。

## P1.5e DIC 派生可靠性（Issue #28）

`p1_dic_reliability_v2.2` 只在 MAB 直接 DIC 锚点上检验无 DIC 标签泄漏的派生链。主链使用 Issue #25 外层交叉拟合 SSS、与相同外层划分匹配的 fCO2 模型、Issue #27 冻结的层次 TA 预测，再由 PyCO2SYS 反演 DIC。比较独立误差、±0.5 有界相关和共享 SSS 扰动的协方差传播；稳定状态用 PyCO2SYS 有限差分 Jacobian，低盐或 Jacobian 无效状态用 2,048 次分块精确 Monte Carlo。256/512/1,024/2,048 draws 的收敛阈值、化学有效范围、A/B/C/D 区间宽度及直接观测门槛均冻结在 `configs/p1_dic_reliability_v2.2.yaml`。

DIC 等级不得高于 SSS、fCO2、TA 中最弱的等级。locked DIC/TA 与 41 个外部 CODAP 航次保持封存；数值闭合只作为工程检查，不能替代直接 DIC 验证。

正式运行匹配到 397 条直接 DIC 锚点、31 个航次、12 年；forward 子集为 113 条、5 个航次、2 年。完整预测链的结果为：

| 外层验证 | RMSE (µmol kg-1) | MAE | bias | R² | common-SSS 90% coverage | 90% width |
|---|---:|---:|---:|---:|---:|---:|
| cruise | 49.61 | 37.70 | -12.78 | 0.180 | 0.912 | 216.2 |
| spatial block | 58.90 | 46.43 | -18.79 | -0.156 | 0.819 | 172.8 |
| subregion | 51.06 | 38.21 | -4.43 | 0.131 | 0.877 | 214.1 |
| forward | 39.53 | 31.57 | -10.37 | 0.265 | 0.956 | 180.6 |

观测 TA oracle 将合并外层 RMSE 从 52.30 降至 28.28；观测 SSS oracle 反而为 58.13，说明 TA 仍是首要瓶颈，且用预测 SSS 训练出的 TA/fCO2 链不能通过事后替换单个盐度输入来改善。54 条、2 个航次的全部观测输入 oracle RMSE 为 11.58，但证据不足以授权产品。稳定域 Jacobian 与低盐/边界精确 Monte Carlo 均完成；最终 1,024→2,048 draws 的中位宽度变化 0.19%，端点变化 1.79 µmol kg-1。数值闭合通过，但空间 R²、覆盖率、区间宽度、A/B 保留面积和上游资格失败。

最终状态为 **`diagnostic_only`**。2025 MAB 共 14,220 个 grid-month，因 Issue #26 的 MAB fCO2 未获区域资格且 Issue #27 TA 为 D，DIC 全部继承为 D 并 suppress；locked 与外部独立标签仍未开启。完整报告、8 张带 caption 的图和 10 个源表见 `docs/experiment_archive/p1_dic_reliability_v2.2/`。

## P2.0 产品范围与证据审计（Issue #36）

`p2_scope_reconciliation_v2.3` 对 #25–#28、#32 的不可变报告和 manifest 做治理审计，没有读取行级标签，也没有打开新的 locked 或 external-independent 数据。冻结结果如下：

| 目标与区域 | 正式状态 | 最高已打开证据 | P2 含义 |
|---|---|---|---|
| SSS，北美相邻 P1 评价域 | `diagnostic_only` | development OOF + 已使用 internal locked | locked 通过 7/8 门槛但 LME 17 失败；旧 locked 已耗尽，全球目标必须建立新盲测 |
| fCO2，Caribbean LME 12 | `pass_regional` | development outer OOF | 只允许区域候选；locked/external 继续封存 |
| fCO2，其他评价 LME | `diagnostic_only` | development outer OOF | 全球 atlas 不构成全球验证 |
| TA，MAB / SAB | `diagnostic_only` / `fail` | development grouped validation | 仅北美可选扩展，预算不超过 P2 的 10% |
| DIC，MAB | `diagnostic_only` | direct DIC development anchors | 上游未同时合格，产品全部 suppress |

产品层级冻结为全球沿海 SSS/fCO2 核心研究线与北美 TA/DIC 非阻断扩展。全局 covariate/atlas 只能证明可计算的投影域，不能证明直接观测覆盖、locked 证据或独立验证。9 个源表、8 张带 caption 的图、上游 SHA256 和机器可读 label-exposure ledger 见 `docs/experiment_archive/p2_scope_reconciliation_v2.3/`。

## P2.1 文献基准与 v1.x 改进合同（Issue #37）

`p2_benchmark_freeze_v2.3` 核对了 ReCAD v1.1、ULB-SOM-FFN-coastal v2、CMEMS-LSCE、RFR-LMEs、区域高分辨率 pCO2、2026 transfer learning、GLORYS、ESA CCI、SMAP 与近期区域 SSS 研究。原生论文指标的目标定义、分辨率、时期、观测支持和验证 split 不同，因此不能直接用于模型选择。

冻结四层面板：publication-native context、0.25° monthly all-support、0.25° monthly common-support，以及相同数据/split/训练折内校准的 strict-method panel。fCO2 的 minimum cruise/spatial/forward ceiling 为 30/40/40 µatm，SSS 为 1.0/1.2/1.2 PSU；competitive 分别为 25/35/35 与 0.8/1.0/1.0，ideal 为 20/30/30 与 0.6/0.8/0.8。每一级还要求相对 strict v1.x/最强 same-split baseline 改善 5%/10%/15%、所有 scheme 正 skill、worst supported-LME 比值≤1.10、校准覆盖率和固定覆盖风险通过。2004–2005 继续只作历史描述。完整报告、11 个源表和 8 张带 caption 的图见 `docs/experiment_archive/p2_benchmark_freeze_v2.3/`。

## P2.3b 全球沿岸入参可用性审计（Issue #52）

`p2_predictor_availability_v2.3` 在 Issue #50 冻结的 400 km 产品掩膜上逐月扫描 1993–2025 的 SST、GLORYS SSS、ADT、CCMP 风速和大气 xCO2，并以球面面积、0–200 m 陆架、10° 纬度带和 Issue #38 SOCAT 行四种口径审计。没有训练模型，没有读取目标值，也没有开启 locked 或 external-independent 标签。

SST、风速和大气 xCO2 的 396 个月覆盖均为 100%；GLORYS SSS 的全球沿岸/陆架面积覆盖为 98.13%/96.39%，归类为 `deployable_with_missingness`。ADT 平均覆盖 92.67%，但最差月全球沿岸仅 87.71%、陆架 80.45%，归类为 `repair_required`。五个动态场严格同月联合覆盖的全球沿岸均值/最差为 92.14%/87.30%，陆架为 89.33%/79.44%。SOCAT 行上的 ADT 覆盖约 97.45%（fCO2）和 99.05%（SSS），明显高估完整产品域，故不得以观测点覆盖代替全球审计。

决策为 `pass_availability_audit`：#53 的必需动态底座为 SST、风速和大气 xCO2；GLORYS SSS 使用显式缺失掩膜；ADT 仅作有回退路径的可选输入。全球 Chl-a 缓存缺失且原始路径当前不可用，MLD 和河流流量没有冻结数据源，均不得成为必需输入。完整报告、11 张带 caption 的图和 12 个源表见 `docs/experiment_archive/p2_predictor_availability_v2.3/`；网格级审计产物保存在 Git 外目录。

## P2.3c 全球沿岸入参立方体（Issue #53）

`p2_global_predictor_cube_v2.3` 将 Issue #52 允许的五个标准化动态场写入 Issue #50 的沿岸节点压缩网格。最终 NetCDF 为 396 月 × 838,661 节点、332,109,756 个 grid-month、42 个变量和 1.571 GB。每个动态场都保存原始可接受值、availability、四级状态码和 source-era；静态部分保存 source-grid 映射、经纬度、面积、距显著陆地距离、GEBCO 水深和陆架/深水分类。

`predictor_ready_mask` 只要求 SST、风速和大气 xCO2，因此 1993–2025 为 100%；GLORYS SSS 作为显式缺失的背景场，ADT 作为不阻断发布的可选场。五场 `strict_common_support_mask` 只用于共同支持消融，面积覆盖均值/最差为 92.14%/87.30%。基础 cube 不做插补、climatology、标准化或任何目标相关变换，这些操作必须在训练折内部拟合。

全部 8 个冻结输入 SHA256 通过；五场 source replay、状态、availability、掩膜恒等式和 Issue #52 的 1,980 条月覆盖重放共 10,492 个检查点零 mismatch。最终状态为 `pass_cube_freeze`，允许 #54 在完全相同 cube、观测与 split 上开始 SSS/fCO2 入参消融。完整报告、11 张带 caption 的图和 12 个源表见 `docs/experiment_archive/p2_global_predictor_cube_v2.3/`；1.571 GB 产品保存在 Git 外目录。

## P2.3d SSS/fCO2 入参消融（Issue #54）

`p2_predictor_ablation_v2.3` 在同一冻结 predictor cube、SOCAT 航次格月和 split 上完成 224 个 CatBoost 拟合。SSS 使用 750,040 条记录、4,615 个航次、54 个 LME；fCO2 使用 887,811 条记录、5,143 个航次、54 个 LME。全部变体共享 cruise、空间块、整 LME 与前向时间留出；可选场缺失不删除样本，2026 provisional、locked test 与 external-independent 标签均未开启。

SSS 的背景场必须按残差订正使用：有 GLORYS SSS 时预测 `SSSobs-SSSbg`，缺失时回退到同折 A2 direct 模型。该分支相对 A2 将 cruise/spatial/whole-LME/2019–2021 forward/2022–2025 forward RMSE 从 0.962/1.058/8.334/1.760/1.956 降至 0.907/0.973/2.470/1.227/1.580 PSU，并通过共同支持与 2,000 次航次级 bootstrap 门槛。ADT 与额外 missing-field dropout 均未通过。

fCO2 的背景 SSS 将 pooled cruise/spatial RMSE 从 36.484/39.001 降至 36.070/38.469 µatm，但 cruise bootstrap 置信区间跨零，不能冻结为生产输入；ADT 同样失败。因此 #40 SSS 冻结为 A2 + masked background-SSS residual/fallback，#41 fCO2 冻结为 A2（时空、沿岸几何、SST、风速、大气 xCO2）。当前点模型仍未满足 Issue #37 的全部最低目标，尤其是 late-forward 和 whole-LME 尾部。完整报告、12 张带 caption 的图和 15 个源表见 `docs/experiment_archive/p2_predictor_ablation_v2.3/`；匹配表和 224 份逐折预测保存在 Git 外目录。

## P2.4 全球沿岸 SSS 模型族比较（Issue #40）

`p2_sss_global_bakeoff_v2.3` 在相同 750,040 条 SOCAT SSS、4,615 个航次、54 个 LME 和 Issue #54 的 16 个冻结输入上比较 raw GLORYS、稳健区域订正、global CatBoost、8 个概率环境/GMM experts、top-2 experts 与单目标 ST96。全部候选使用相同 cruise、空间块、整 LME和前向时间外层 split，共生成 96 份预测分区和 32 个模型 checkpoint；旧 #25 locked、2026 provisional 与 external-independent 标签均未开启。

GMM experts 的 pooled cruise/spatial/forward RMSE 为 0.864/0.918/1.422 PSU，优于 CatBoost 的 0.907/0.973/1.446，且两种普通外层 split 的 2,000 次航次 bootstrap MSE 差均显著低于零。冻结的主指标 cruise-equal RMSE 从 1.049 降至 1.000 PSU，改善 4.61%，未达到 5% 门槛；whole-LME 从 2.470 恶化到 2.620 PSU，2022–2025 forward 仍为 1.560 PSU。LME 17 的 spatial/whole-LME/forward RMSE 为 5.584/6.136/7.104 PSU，均未改善。ST96 与 top-2 experts 在普通 split 和 whole-LME 上均不支持升级。

最终状态为 **`diagnostic_only` / no promotion**：保留 global CatBoost 作为可重复基线，GMM 只作为显式观测支持域内的 challenger；不授权无条件全球 SSS 订正产品。经验区间的 forward 90% coverage 为 CatBoost 0.876、GMM 0.873，仍不足以替代新盲测和外部独立验证。完整报告、12 张带 caption 的图和 15 个源表见 `docs/experiment_archive/p2_sss_global_bakeoff_v2.3/`；预测、checkpoint 和分析中间表保存在 Git 外目录。

## P2.5 全球沿岸 fCO2 模型族与区域共享比较（Issue #41）

`p2_fco2_global_bakeoff_v2.3` 在相同 887,811 条 SOCAT fCO2、5,143 个航次、54 个 LME 和 Issue #54 的冻结 A2 输入上比较 seasonal/trend background、诚实重训 v1.x Random Forest、direct CatBoost、两种 residual CatBoost、固定 LME adapter 与 8 个概率环境/GMM experts。每个外层折都重新构造 SSS：外层训练部分采用两折 cruise cross-fit，held 部分只由全部外层训练 SSS 标签拟合；observed SSS 从未直接作为 fCO2 输入。最终保存 16 份 cross-fitted SSS、112 份预测和 176 个模型 checkpoint。

诚实 RF 在普通外推中仍最强，pooled cruise/spatial RMSE 为 33.96/37.45 µatm；所有现代候选相对 RF 的 cruise 与 spatial 航次级 paired bootstrap MSE 差均显著大于零。LME adapter/GMM 将 forward RMSE 从 47.55 降至 46.21/46.45 µatm，且 GMM 的 cruise 年 bias slope 为 -0.013 µatm yr-1，优于 RF 的 -0.170，说明显式 trend/residual 分支更能跟踪时间变化；但 whole-LME、普通 cruise 和既有 Caribbean 区域技能同时退化。2022–2025 GMM RMSE 仍为 46.58 µatm，高于 40 µatm 门槛。

Cross-fitted SSS 对 residual CatBoost 的 cruise/spatial/whole-LME/forward RMSE 产生一致但有限的改善：36.93/39.20/57.43/47.46 降为 36.45/38.58/57.28/46.57 µatm。GMM 在简单的绝对 RMSE/正 skill 条件下有 22 个 development-qualified LME，RF 为 18 个；但 GMM forward 50/90 coverage 只有 0.396/0.806，不能把该计数解释为产品可信覆盖。

最终状态为 **`diagnostic_only` / no promotion**：诚实 v1.x RF 保留为 strict interpolation baseline，residual/GMM 只作为 #42 temporal/connectivity 实验的时间迁移分支。下一候选必须同时保留 RF 的 cruise/spatial 技能和 residual 模型的趋势/late-forward 优势。locked、external-independent 与 2026 provisional 标签均未开启。完整报告、15 张带 caption 的图和 22 个源表见 `docs/experiment_archive/p2_fco2_global_bakeoff_v2.3/`；大文件保存在 Git 外目录。

## P2.6 时间上下文与沿岸连通性（Issue #42）

`p2_context_connectivity_v2.3` 在 Issue #41 的相同 887,811 条 fCO2、5,143 个航次、54 个 LME、strict cross-fitted SSS 与 16 个冻结外层折上比较诚实 v1.x Random Forest、6 个月 lag CatBoost、两层 GRU、单目标 ST96 和四邻接水体图 ST96。时间序列只读取 observation-independent predictor cube 的 SST、风速、大气 xCO2 和背景 SSS；没有使用历史 fCO2 或 observed SSS。RF 相邻水体网格残差相关为 0.869、同月置换对照约为零，因此图分支按预设条件触发。

RF 的 pooled cruise/spatial/whole-LME/forward RMSE 为 33.96/37.45/54.38/47.55 µatm。Lag CatBoost 只把 forward 降到 47.16，并将 bias 从 -15.06 降到 -5.14 µatm，但 cruise/spatial/whole-LME 的 cruise-equal RMSE 分别退化 7.5%/7.2%/4.7%。GRU、局地 ST96 和图 ST96 在所有外层方案均明显退化；图 ST96 虽相对局地 ST96 改善 cruise/forward，却未消除相邻残差结构。所有 context candidate 的 cruise/spatial paired bootstrap MSE 差均显著大于 RF，forward 区间校准也未通过。

嵌套 LME 内 cruise 学习曲线进一步区分数据与模型容量：训练 cruise 从 429 增至 4,161 时，RF fold-0 RMSE 从 41.57 持续降至 32.49；lag CatBoost 在约 40% 后停滞于 35.9；ST96 始终高于 41 且不稳定。结果支持“当前独立航次覆盖与月尺度入参限制有效模型复杂度”，但不支持“单纯增加相同数据量即可释放 Transformer”。最终状态为 **`stop_scaling_context_family` / no promotion**；保留 RF point baseline，禁止将不连续 LME-month background 原样用于连续产品。完整报告、17 张带 caption 的图和 22 个源表见 `docs/experiment_archive/p2_context_connectivity_v2.3/`；80 份外层预测、18 份学习曲线预测、82 个 checkpoint 与上下文缓存保存在 Git 外目录。

## P2.7 证据审计与路线修订（Issues #43、#61–#64）

2026-10-13 只读审计核对了 v2.2 split manifest、v2.3 global group manifest、#54 matchup 元数据以及 #40–#42 runner/analyzer。旧 SOCAT locked 的 538/538 航次进入 v2.3 manifest，其中 417 航次、79,135 行进入实际 matchup；旧 SOCAT external 的 9/9 航次进入 manifest，其中 7 航次、1,474 行进入 matchup。该核对没有读取新的目标列，但足以证明相关组已参与开发数据链，旧“sealed”身份不可恢复。

空间审计显示，各折 held 行中 85.2%–92.0% 的精确 1/8°格点也出现在训练，绝大多数实际 5°块同样重叠。现有结果继续表示新航次分组压力测试，不表示真正新地点外推。校准审计还发现其他 outer-fold 残差会由训练过当前 held 标签的模型产生；forward 仅筛早期残差行不能消除生成模型使用未来标签的问题。#40 的 `salinity_band` 由 `truth` 构造并用于区间宽度，只能保留为事后误差分层。

路线因此调整为：#61 先冻结暴露账、真实 geographic block + buffer、past-only rolling forward、outer-training 内 fit/selection/calibration 和更正基线；#62 再以相同小型 inner-search 预算比较 strict-v1 RF、modernized RF、ExtraTrees、boosting、MLP/ResNet、表格 Transformer 及必要时序候选；#63 在 #61 后测试 Chl-a、MLD、淡水/环流代理和连续背景；#64 只在 inner-OOF 残差互补或低支持失败明确时测试集成、概率头和预训练。#44、#5、#46 依次后移。

#42 的有效结论收窄为“停止放大当前 GRU/ST96/四邻接实现”。其短训练、单种子、残差背景和简化邻域输入不足以否定全部 Transformer、时序或图模型。RF 暂作为 modernized point baseline；exact strict-v1 replica 在 #61/#62 中单独实现与命名。

## P2.7a 验证、暴露谱系与基线修复（Issue #61）

`p2_validation_repair_v2.4` 在冻结的 1993–2025 matchup 上建立了可复用的真实空间和时间验证合同。SSS 使用 750,040 条观测，fCO2 使用 887,811 条观测；两者均运行 5 个完整航次 folds、5 个真实 5° geographic folds（移除 held 航次并增加 100 km 缓冲）、5 个完整 LME/neutral-region folds 和 3 个严格 past-only forward windows。每个 outer training 再按完整 group 切为 80% 模型拟合与 20% 区间校准；全部 group、record、fit-calibration group 和 forbidden-future overlap 为零。旧 SOCAT locked/external 与 carbon-product 别名交集已写入机器可读 exposure ledger，不再拥有 blind 身份。

SSS CatBoost 背景残差订正在 cruise/geographic/whole-region/forward 的 RMSE 为 0.921/1.593/2.571/1.284 PSU；raw GLORYS 为 1.721/1.721/1.721/1.871 PSU。因此订正在普通航次与时间外推上有效，但完整区域迁移会明显劣于背景场，不能作为无条件全球订正。fCO2 modernized RF 的 cruise/forward RMSE 为 34.38/40.54 µatm，direct CatBoost 的 geographic/whole-region RMSE 最低但仍为 52.12/53.18 µatm；strict-v1 RF 的 geographic/whole-region 为 62.53/66.81 µatm。所有 fCO2 绝对 RMSE promotion gates 均失败。

训练内嵌套区间在 cruise 上接近 nominal，但在 geographic、whole-region 和 forward 常出现 under-coverage，故 interval width 只保留为风险排序诊断，不能称为已校准产品误差。最终状态为 **`corrected_development_baselines_only`**：#61 的验证修复 gate 通过并允许 #62 使用相同 folds 做公平 model-zoo；没有 SSS/fCO2 模型获得生产提升。完整报告、12 张 caption 紧随图片的图、9 个源表和外部大文件哈希见 `docs/experiment_archive/p2_validation_repair_v2.4/`。

## P2.7b 公平有限模型族筛选（Issue #62）

`p2_model_zoo_v2.4` 在 Issue #61 的相同 750,040 条 SSS、887,811 条 fCO2 development 观测和修复后的 cruise、真实 5° geographic + 100 km buffer、whole-region、past-only forward folds 上，公平比较 Ridge、ExtraTrees、CatBoost、XGBoost、Residual MLP 与紧凑 FT-Transformer。每个模型族使用 6 个冻结 inner-group 配置、相同训练/选择行上限和 outer-training 内独立校准；fCO2 只使用 outer-nested predicted SSS。初筛 216 个分区，预注册 finalist 再增加两种子，最终保存 324 个 OOF 分区、936 个 checkpoint 文件和完整资源账。没有打开新的 locked、external-independent 或 2026 标签。

SSS CatBoost 是唯一通过“至少两个 scheme 改善 5%，其他 scheme 不退化超过 3%”相对门槛的 finalist；三种子 cruise/geographic/whole-region/forward RMSE 为 0.899/1.436/2.168/1.186 PSU。但是 geographic 仍高于 Issue #37 的 1.2 PSU ceiling，geographic/whole-region 的 90% coverage 只有 0.727/0.621，因此不满足完整生产 gate。ExtraTrees 与 XGBoost 的普通 cruise/forward 改善伴随 geographic/whole-region 明显退化；Residual MLP、FT-Transformer 和 Ridge 没有提供可迁移优势。

fCO2 没有模型通过相对 promotion 门槛。ExtraTrees 与 CatBoost 仅作为三种子稳定性 finalists：前者 RMSE 为 33.301/51.242/52.408/39.228 µatm，后者为 34.907/48.950/53.092/40.079 µatm；两者均未满足 30/40/40 µatm 的 cruise/geographic/forward 合同，shifted interval 也 under-cover。相邻水体同月残差相对置换对照的相关增量为 SSS 0.847、fCO2 0.845，超过冻结的 0.20 graph trigger；temporal trigger 对两目标均未触发。该信号只授权后续受控图假设，不允许继续放大 #42 已失败的四邻接 ST96 实现。

最终状态为 **`development_model_zoo_complete` / no production promotion**：SSS 保留 CatBoost 开发 finalist；fCO2 保留 ExtraTrees/CatBoost stability finalists；当前数据下更高点模型容量没有解决区域迁移。完整报告、13 张 caption 紧随图片的图和 15 个源表见 `docs/experiment_archive/p2_model_zoo_v2.4/`；逐行预测、配置记录和 checkpoint 保存在 Git 外目录 `C:\backup\phd\data\processed\recad_v2_4\p2_model_zoo_v2.4`。

## P2.7c 过程变量与连续背景（Issue #63）

`p2_process_predictors_v2.4` 在 Issue #61 的相同修复 folds 与 Issue #62 的目标特定树模型上，逐一测试同月水体四邻接 SST/风速锋面、背景 SSS 淡水亏损与局地梯度、可选 ADT 环流梯度，以及 fCO2 的训练折内连续球面-季节-趋势 Ridge 背景残差。全部可选场采用显式 availability/neighbour count 与保留全样本的回退；同时报告 all-support 与 common-support。Chl-a 原路径不可用，MLD 和河流流量没有冻结源，因此作为不可用负结果归档，没有伪造同名代理。没有开启新的 locked、external-independent 或 2026 标签。

SSS 三个过程代理均未通过冻结门槛：锋面和 ADT 只在 whole-region 单一 scheme 有约 3.0%/3.6% 改善，却在 cruise 或 geographic 退化；背景 SSS 淡水代理令 geographic/whole-region 更差。fCO2 的三个过程代理同样失败；其中淡水代理令 geographic RMSE 退化 4.2%。唯一保留的是连续背景残差，其三种子 cruise/geographic/whole-region/forward RMSE 为 32.760/54.201/52.754/39.263 µatm，对匹配 ExtraTrees 的改善很小，且 90% coverage 仍只有 0.796/0.782/0.837（geographic/whole-region/forward）。

最终状态为 **`retain_fco2_continuous_background_for_development_only` / no production promotion**。#64 不获得过程因子组合授权；只允许针对 #62 已触发的残差图结构测试改变架构或信息路径的受控假设，以及针对空间/时间 shift under-coverage 的不确定性假设，禁止重跑 #42 的四邻接 ST96。完整报告、10 张 caption 紧随图片的图和 11 个源表见 `docs/experiment_archive/p2_process_predictors_v2.4/`；270 个 OOF 分区、特征缓存和 1,080 个 checkpoint 文件保存在 Git 外目录。

## P2.7d 条件图、集成与不确定性（Issue #64）

`p2_conditional_combination_v2.4` 在 Issue #61 的完全相同 corrected folds 上测试三个被前序证据触发或允许筛选的分支。季节水体图分支只传播由 inner-training 模型在 honest inner-validation 行产生的残差，改变了 #42 的架构和信息路径；不确定性分支用同一 honest residual 拟合 log-absolute-error ExtraTrees scale head，再由 outer-training calibration 做标准化 conformal。过程因子组合与预训练没有触发，均作为负结果保留。

图残差 adapter 对 SSS 只有 pooled cruise RMSE 从 0.895 到 0.873 的明显改善，其他迁移方案不足 1%；fCO2 cruise 和 forward 还会退化，两个目标均未通过两方案 paired-group 门槛。scale head 虽将 fCO2 geographic/forward 90% coverage 从 0.795/0.837 提高到 0.833/0.847，但仍低于 0.85，并把 whole-region 从 0.783 降到 0.742，因此不能取代产品 reliability flag。

honest inner-OOF 的 fCO2 CatBoost/F4 残差中位相关为 0.860，触发冻结凸组合。逐折权重完全由 inner labels 决定；outer pooled cruise/geographic/whole-region/forward RMSE 从 32.776/54.061/52.834/39.317 降至 **32.471/52.495/52.504/38.853 µatm**。geographic 和 forward 改善超过 1%，paired grouped bootstrap 上界低于零，且无方案退化，故最终状态为 **`retain_fco2_convex_ensemble_for_development_only` / no production promotion**。完整报告、8 张 caption 紧随图片的图和源表见 `docs/experiment_archive/p2_conditional_combination_v2.4/`；未开启新 locked、external-independent 或 2026 标签。

## P2.8 全球校准与适用域合同（Issue #44）

`p2_global_reliability_contract_v2.4` 在 Issue #61 corrected folds 上为 Issue #62 的 SSS CatBoost 背景残差 finalist 和 Issue #64 的 fCO2 CatBoost/F4 凸组合建立嵌套区间、风险排序、A/B/C/D、trusted/diagnostic/suppressed 状态与 reason bits。SSS/fCO2 cruise RMSE 为 0.896 PSU / 32.42 µatm，90% coverage 为 0.898/0.901；geographic、whole-region、forward 中最低 coverage 为 0.788，未通过冻结的 0.85 shifted gate。

最终写出两个完整 1993–2025 NetCDF，各含 396×838,661 个 grid-month。平均面积加权 A/B 为 SSS 20.61%、fCO2 20.97%，D/suppressed 为 70.47%/69.11%，没有通过删除困难区域制造结果。结论为 **`freeze_applicability_contract_but_fail_global_interval_qualification`**：flagged 产品和校准对象可进入一次性独立评估准备，但不允许宣称无观测沿岸已获全球准确度资格。完整报告、12 张 caption 紧随图片的图和 14 个源表见 `docs/experiment_archive/p2_global_reliability_contract_v2.4/`；大文件和 SHA256 保存在 Git 外目录。

## P2.X 北美 TA/DIC 可选分支触发审计（Issue #45）

`p2_na_carbonate_optional_v2.4` 核对了冻结的 entry condition：只有存在已合格的北美 SSS/fCO2 部署输入，才允许消耗最多 10% 的 P2 预算测试 TA/DIC rescue。Issue #44 冻结了 development support grades，但 shifted 90% coverage 最低为 0.788，明确没有取得独立产品资格；MAB TA/DIC 仍为 `diagnostic_only`，SAB TA 仍为 `fail`。因此该分支以 **`untriggered_non_blocking`** 关闭，没有训练新模型、派生 DIC、消耗预算或开启 locked/external-independent/2026 标签。完整触发审计和哈希证据见 `docs/experiment_archive/p2_na_carbonate_optional_v2.4/`。

## P3 一次性独立评估冻结准备（Issue #5）

`p3_independent_freeze_v2.4` 已冻结 SSS Issue-62 CatBoost background-residual 三种子 ensemble、fCO2 Issue-64 CatBoost/F4 convex ensemble、Issue-44 calibration/grade/mask、21 个 checkpoint/config artifact hashes、predictor cube/coastal mask、指标与报告规则，并准备 Issue #46 conditional release schema。metadata-only 审计只读取文件名、版本和 Issue-61 exposure ledger，没有读取任何新 target label。

当前状态为 **`prepared_waiting_for_eligible_independent_set`**。本机只有 SOCAT v2026；旧 locked 538/538 航次组与 old external 9/9 组均进入 P2 manifest，其中 417/7 组进入实际 matchup，永久不能恢复 blind 身份。未来 post-v2026 SOCAT exact set 尚不存在，因此 `label_access_authorized=false`、access count=0。完整 lock、协议、源表和阻塞条件见 `configs/frozen/p3_lock_v2.4.json` 与 `docs/experiment_archive/p3_independent_freeze_v2.4/`。

用户随后确认 SOCAT v2026 已是最新且没有其他可用独立来源，因此 P3 最终状态更新为 **`terminal_no_eligible_independent_source`**。没有执行开盲，access count 仍为 0。SSS/fCO2 的 1993-2025 完整场保留为带适用域和不确定性的 `diagnostic_only` candidate；没有 trusted target-region release。Issue #46 同步关闭为 untriggered，TA/DIC 继续 suppressed。终端证据边界见 `docs/experiment_archive/p3_independent_unavailable_v2.4/`。

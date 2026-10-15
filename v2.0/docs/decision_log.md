# ReCAD v2 决策日志

本日志只记录改变研究方向、验证口径或产品定义的决定。实验数值见 `experiment_registry.md`，数据版本见 `data_registry.md`。

| ID | 日期 | 决定 | 依据 | 状态/替代关系 |
|---|---|---|---|---|
| D001 | 2026-09-05 | 正式结论必须来自 grouped out-of-sample 和外部验证；随机点切分不足 | 沿航次和空间自相关会导致泄漏 | 当前有效 |
| D002 | 2026-09-06 | MODIS Chl-a 作为 fCO2 输入保留，但必须显式处理 2002–2020 之外缺失 | 共同覆盖期三种子 dev RMSE 中位数改善 6.28% | 当前有效；尚未全球构建 |
| D003 | 2026-09-07 | CO2SYS 软约束不能替代 TA/DIC 观测；不以闭合误差单独选择模型 | B1/B2 闭合改善约 92%，预测技能只改善约 0–3% | 替代“加大化学权重即可解决” |
| D004 | 2026-09-09 | 主结构只学习 SSS/fCO2/TA；DIC 由 inverse CO2SYS 派生并用直接 DIC 监督/验证 | 碳系统自由度分析与 emulator RMSE 0.552 µmol/kg | 当前有效 |
| D005 | 2026-09-09 | 美国西岸不得因随机 test 较好进入生产 | 留航次 TA RMSE 约 75–127，随机 test 明显乐观 | 当前有效 |
| D006 | 2026-09-28 | 产品拆成全球 SSS/fCO2 主产品和北美 TA/DIC 扩展；全球无 TA 支持区不发布 DIC | 全球 fCO2/碳参数覆盖与区域实验审计 | 当前产品定义 |
| D007 | 2026-09-28 | 2004–2005 降级为历史 benchmark | 已被多轮模型选择和结果查看 | 不可恢复为独立验证 |
| D008 | 2026-09-28 | CODAP 与 GLODAP 先保留逐观测 provenance 并去重，再建立 split；不先月平均 | 74 个精确 EXPOCODE 重叠且存在航次别名 | 当前有效 |
| D009 | 2026-09-28 | 不采用硬 N 区域独立生产模型；比较全局主干、区域平衡、软门控 residual experts 和 coastal graph | 统一模型受密集区支配，硬省界产生人工跳变 | P1/P2 待验证 |
| D010 | 2026-09-28 | 模型训练分阶段，最终联合概率推理；SSS/fCO2/TA 误差样本经 CO2SYS 传播到 DIC | 避免点预测串联误差和四自由头不可识别 | 当前结构候选 |
| D011 | 2026-09-28 | GitHub #1 为父 Roadmap；#2–#5 分别管理 P0–P3 并作为正式 sub-issues；实验方法/数据/结果由仓库总账记录 | 防止任务和独立验证口径漂移 | 当前治理规则 |
| D012 | 2026-10-04 | fCO2 当前只作为 Caribbean Sea（LME 12）区域产品候选；其他 LME 输出背景回退并标 D，锁定集继续封存 | #26 中 LME 12 的 cruise/空间块/前向 RMSE 为 9.54/10.03/13.57 µatm 且全部正 skill；LME 6 空间确认失败，整 LME 转移不成立 | `pass_regional`；待未来一次性 locked fCO2 audit |
| D013 | 2026-10-08 | MAB DIC 保持派生诊断，不发布 2025 产品值；只有 fCO2 与 TA 在同一网格月均获资格后才重新开启 DIC 产品门槛 | #28 完整链 cruise/spatial/subregion/forward RMSE 为 49.61/58.90/51.06/39.53 µmol kg-1；空间 R²=-0.156，区间门槛失败，观测 TA oracle 将合并 RMSE 从 52.30 降至 28.28；14,220 个 2025 MAB grid-month 全部继承 D | `diagnostic_only`；locked/external 继续封存 |
| D014 | 2026-10-08 | P2 冻结为全球沿海 SSS/fCO2 核心研究线；北美 TA/DIC 是预算不超过 10% 的非阻断可选扩展；atlas 投影域与直接观测验证域必须分开陈述 | #25 正式 SSS 决定为 `diagnostic_only`，已用 locked audit 失败 LME 17；#26 仅 LME 12 为 `pass_regional`；#27/#28 MAB TA/DIC 均为 `diagnostic_only`；Issue #36 证据与标签暴露审计 | 当前有效；旧 SSS locked 永久不得再次作为 blind，全球资格只能由新全球观测和新盲测取得 |
| D015 | 2026-10-08 | 冻结四层基准面板和 v1.x 改进合同；论文原生指标只作背景，模型选择只用相同观测、split、校准和输入的 strict-method panel | #32 证明 v1.1 公开分数受全期局地校准影响；2024–2026 沿海 pCO2 与 SSS 文献的目标、区域和验证设计不可直接排名；Issue #37 归档 | 当前有效；最低 tier 要求至少 5% paired 改善、绝对 RMSE、区域稳健性、覆盖率和固定覆盖风险同时通过 |
| D016 | 2026-10-12 | P2.3 全球立方体以 SST、风速和大气 xCO2 为必需动态场；GLORYS SSS 带显式缺失掩膜；ADT 只能作为有回退路径的可选场，不能决定网格月是否发布 | Issue #52 在 1993–2025 冻结沿岸掩膜上测得 GLORYS SSS 最差月面积覆盖 98.13%，ADT 为 87.71%、陆架为 80.45%；五场严格同月联合覆盖均值/最差为 92.14%/87.30% | 当前有效；Chl-a、MLD、河流流量不得作为 #53 必需输入 |
| D017 | 2026-10-12 | 冻结 1993–2025 沿岸节点压缩 predictor cube；基础 cube 保留 NaN 和逐变量状态，不做全期插补、climatology 或标准化；训练相关变换必须在各训练折内拟合 | Issue #53 生成 396×838,661、42 变量的 1.571 GB NetCDF；10,492 个重放/恒等式检查零 mismatch；mandatory predictor-ready 为 100%，五场 strict support 均值/最差为 92.14%/87.30% | 当前有效；Issue #54 及后续模型必须使用 SHA256 `0f4e9f37…d0401` 或显式提升版本 |
| D018 | 2026-10-12 | #40 SSS 使用 A2 必需输入加 masked GLORYS-SSS 残差订正与同折 direct fallback；#41 fCO2 只使用 A2，不加入背景 SSS 或 ADT | Issue #54 的 SSS 背景残差分支在 cruise/spatial/whole-LME/两段 forward 均改善并通过共同支持和 grouped bootstrap；fCO2 背景 SSS 的 pooled 改善约 1%，但 cruise bootstrap 跨零；ADT 对两个目标均失败 | 当前有效；可选输入不能因 pooled 改善或覆盖筛选单独进入生产；最终模型仍需 #40/#41 改善 forward 与 whole-LME 尾部 |
| D019 | 2026-10-12 | #40 不提升任何 SSS 候选为无条件全球生产模型；global CatBoost 保留为冻结基线，GMM experts 只在后续显式支持域内作为 challenger | GMM pooled cruise/spatial/forward RMSE 改善到 0.864/0.918/1.422 PSU，但主指标 cruise-equal 改善仅 4.61%，whole-LME 恶化到 2.620 PSU，LME 17 和 late-forward 失败；ST96/top-2 experts 更差 | `diagnostic_only`；下一阶段必须以支持感知 gate、背景回退和新盲测解决可发布范围，不能以更大模型替代迁移证据 |
| D020 | 2026-10-13 | #41 不提升任何 fCO2 候选；诚实 v1.x Random Forest 保留为 strict interpolation baseline，residual/GMM sharing 只进入 #42 temporal/connectivity 对照 | RF 的 cruise/spatial RMSE 33.96/37.45 µatm 为最优且所有现代候选 paired bootstrap 均退化；LME adapter/GMM 的 forward 46.21/46.45 优于 RF 47.55并减少趋势偏差，但 whole-LME、Caribbean、校准和40 µatm forward 门槛失败 | `diagnostic_only`；下一候选必须同时保留 RF 的普通空间技能与 residual 分支的趋势/late-forward 优势，不能用 development-qualified LME 数量代替校准证据 |
| D021 | 2026-10-13 | 停止扩大 #42 当前 6 个月 GRU/ST96/四邻接实现；保留 modernized RF 作为 fCO2 点插值开发基线，并禁止将离散 LME-month 背景原样用于连续产品 | #42 中 lag CatBoost 仅将 forward RMSE 47.55 降至 47.16 µatm，却令 cruise/spatial/whole-LME cruise-equal RMSE 退化 7.5%/7.2%/4.7%；GRU/ST96 全面退化；residual 模型产生约 20 µatm cross-LME 人工跳变 | `stop_scaling_exact_context_implementation` / no promotion；该结果不否定所有 Transformer、时序网络或图模型，家族结论由修复验证后的 #62 决定 |
| D022 | 2026-10-13 | P2 顺序改为 #61 验证/暴露修复 → #62 公平 model-zoo，并在 #61 后允许 #63 过程变量/连续背景准备；#64 集成与预训练仅在修复证据触发后执行，随后才进入 #44/#5/#46 | 元数据交集显示旧 locked 538/538 航次均进入 v2.3 manifest，其中 417 航次、79,135 行进入 matchup；旧 external 9/9 航次进入 manifest，其中 7 航次、1,474 行进入 matchup。现行 spatial fold 中 85.2%–92.0% held 1/8°格点也见于训练；SSS 区间依赖真值分箱；#40–#42 校准非 outer-training 内嵌套 | 当前有效；#40–#42 点预测只作开发证据，旧 sealed/strict-spatial/leakage-safe uncertainty 表述由 #61 更正，已暴露标签永久不得再作 blind |
| D023 | 2026-10-13 | 接受 #61 修复后的验证合同，但不提升任何 SSS/fCO2 基线；#62 必须复用相同 geographic + 100 km buffer、past-only forward 和 outer-training 内嵌套校准 | 零 group/record/time/calibration leakage；SSS CatBoost cruise/geographic/whole-region RMSE 0.921/1.593/2.571 PSU；fCO2 最佳 cruise/geographic RMSE 34.38/52.12 µatm；strict-transfer 区间多次 under-cover | `corrected_development_baselines_only`；普通 cruise 技能不能替代未见区域资格，旧 #40–#42 uncertainty 与 strict-spatial 表述由不可变 correction notes 收窄 |
| D024 | 2026-10-14 | #62 完成有限公平 model-zoo 后不提升生产模型；SSS 保留 CatBoost 开发 finalist，fCO2 只保留 ExtraTrees/CatBoost stability finalists；停止在当前点输入上扩大 MLP/FT-Transformer | SSS CatBoost 三种子 cruise/geographic/whole-region/forward RMSE 0.899/1.436/2.168/1.186 PSU，仍失败 geographic ceiling 与 shifted coverage；fCO2 ExtraTrees 为 33.301/51.242/52.408/39.228 µatm，没有模型通过完整门槛；两目标 graph signal 0.847/0.845 触发但 temporal 未触发 | `development_model_zoo_complete` / no promotion；#63 优先补过程信息与连续背景，后续图分支必须改变信息或架构并显式对照，不得重跑 #42 已失败实现 |
| D025 | 2026-10-15 | #63 仅保留 fCO2 连续球面-季节-趋势背景残差作为 development candidate；SSS 不增加过程代理，SST/风速锋面、背景 SSS 淡水和 ADT 环流代理均不进入产品 | 连续背景三种子 cruise/geographic/whole-region/forward RMSE 为 32.760/54.201/52.754/39.263 µatm，相对匹配 ExtraTrees 仅小幅改善且 geographic/whole-region/forward coverage 为 0.796/0.782/0.837；SSS 三个代理没有通过两 scheme 成对门槛 | `retain_fco2_continuous_background_for_development_only` / no production promotion；#64 无过程因子组合分支，只允许改变架构/信息的受控图假设和针对 shifted under-coverage 的不确定性假设 |
| D026 | 2026-10-15 | 冻结 #44 的完整域 A/B/C/D、trusted/diagnostic/suppressed、reason-bit 与 abstention 合同，但不宣称全球区间已合格 | Cruise 90% coverage 为 SSS 0.898、fCO2 0.901；geographic/whole-region/forward 最低 coverage 为 0.788；完整 1993–2025 产品中平均 A/B 面积为 20.61%/20.97%，D 为 70.47%/69.11% | `freeze_applicability_contract_but_fail_global_interval_qualification`；保留完整 flagged 产品进入独立评估准备，禁止把无观测海域 A/B 当作已验证准确度 |
| D027 | 2026-10-15 | #45 北美 TA/DIC 可选 rescue 不触发；不以 Issue #44 的 development A/B grades 替代 qualified North-American SSS/fCO2 inputs entry condition | Issue #44 shifted 90% coverage 最低 0.788，未通过 0.85 gate；MAB TA/DIC 仍为 `diagnostic_only`，SAB TA 为 `fail` | `untriggered_non_blocking`；零训练预算、零新标签访问，未来只有新冻结路线提供 independently qualified 上游输入才可重开 |
| D028 | 2026-10-15 | 冻结 P3 SSS/fCO2 candidates、checkpoints、calibration、mask、metrics 与 conditional release schema，但不打开标签 | 本机只有 SOCAT v2026；旧 locked 538/538 和 old external 9/9 groups 已进入 P2 manifest，分别有 417/7 groups 进入 matchup；不存在 post-v2026 exact candidate set | `prepared_waiting_for_eligible_independent_set`；label access 未授权且计数为 0，需 post-v2026 SOCAT 或另一真正独立全球沿岸来源 |
| D029 | 2026-10-15 | P3 改为 SOCAT v2026 内 metadata-only 选择完整航次并在所有拟合/校准路径中整组剔除；不要求或声称跨数据集推广 | 当前科学问题是模型对未见 cruise、稀疏位置和新近时间的可靠性；另一个数据源不能直接回答该问题。冻结 50 条完整航次，覆盖 5 个 basin、23 条 recent 与 27 条 sparse-5° 航次 | D028 的候选/代码哈希继续有效，但“必须等待新数据源”被本决定替代；合入 manifest 后一次性运行同源整航次验证，结论仅限 same-source transfer |
| D030 | 2026-10-15 | P3 通过带 flag 的同源整航次门槛，但按 target/region/grade 限制发布：SSS A/B qualified；fCO2 A/B 仅 Pacific 外 qualified；C diagnostic，D suppressed | 50 条完整航次上 SSS/fCO2 RMSE 1.185 PSU / 30.780 µatm、coverage90 0.910/0.883；A/B 为 0.598 PSU / 26.619 µatm、coverage 0.927/0.904；Pacific fCO2 A/B coverage=0.845 低于 0.85 gate；极端淡水 cruise 被 D 捕获但误差仍巨大 | `pass_flagged_same_source_release_with_regional_suppression`；允许 #46 仅构建这些 qualified 层，不得声称 cross-dataset 或无条件全球可靠性 |

## 被明确否定或限制的旧做法

- 不再把四个碳/盐度目标的平均分作为唯一模型选择依据。
- 不再把 TAest 当作真值，也不在无 TA/DIC 支持区施加强 CO2SYS 约束来“创造”信息。
- 不再把随机 observation-row test 称为独立验证。
- 不再以增加 Transformer 参数量作为默认下一步；先解决分布偏移、采样权重和区域过程差异。
- 不直接拼接 CODAP 与 GLODAP，也不删除重复来源行；使用 `duplicate_group` 和 `is_primary`。
- 不使用错误 SAL 路径的西岸缓存或存在 SSS 目标泄漏的 `weighted_global_v1`。

## 待决策事项

- 外部独立验证的具体平台/航次及其标签隔离方式。
- 正式产品截止 2024 还是 2025，以及未来年份更新协议。
- 全球 Chl-a 缺失期采用双模型、显式缺失 token 还是独立数据源。
- P1 后是否有足够证据升级到 soft MoE；P2 不预先假定复杂结构一定优于区域平衡基线。
- 月平均 CO2SYS 聚合偏差是否需要亚月尺度修正。

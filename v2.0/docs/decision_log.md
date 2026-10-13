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

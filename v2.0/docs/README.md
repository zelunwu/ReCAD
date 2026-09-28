# ReCAD v2 文档入口

本目录按“当前事实只有一个权威入口”的原则维护。旧报告不会删除，但它们只作为实验附件；当前结论、数据版本和下一步不得从旧报告单独推断。

## 四个权威入口

| 文件 | 回答的问题 | 更新时机 |
|---|---|---|
| `data_registry.md` | 有哪些原始/处理数据，版本、路径、哈希、QC 和限制是什么 | 下载、修正、合并或冻结数据时 |
| `experiment_registry.md` | 做过哪些实验，方法、数据、split、结果和证据等级是什么 | 每次正式实验开始和结束时 |
| `decision_log.md` | 为什么改变产品范围、模型结构或验证口径；哪些旧决定已被替代 | 发生科学或工程决策时 |
| `current_status_and_roadmap_20260928.md` | 当前计划、P0–P3 依赖和通过门槛是什么 | Roadmap 变化时 |

共同实验规范在 `experiment_protocol.md`。操作交接和本机路径在 `../HANDOFF.md`。GitHub 任务状态在 Roadmap issue #1；Issue 不是实验结果数据库。

正式实验的 reviewer-ready 证据归档位于 `experiment_archive/<experiment_id>/`。每个归档包含完整报告、可直接回复审稿人的图片、图片源 CSV 和可自动核验的哈希清单；索引见 `experiment_archive/README.md`。

## 当前有效结论

- 全球产品重点为 SSS/fCO2；北美扩展为 TA 和由 inverse CO2SYS 派生的 DIC。
- 2004–2005 已被多轮查看，只是历史 benchmark。
- `weighted_global_v2` 是修复 SSS 泄漏后的全球基线；全球 fCO2 尚未通过产品门槛。
- CODAP+GLODAP 北美表层逐观测产品已完成，正式模型尚未使用新合并数据重训。
- P0 已通过 v2.2 更正：数据轴保留到 2026，SSS/fCO2 核心期到 2025，TA/DIC 观测到 2024 并允许模型前向输出到 2026；下一步进入 P1 单任务基线。

## 详细报告索引

下列文件保留原实验语境和完整解释，但其结论必须以 `experiment_registry.md` 中的证据等级和当前状态为准：

| 报告 | 内容 | 当前角色 |
|---|---|---|
| `p0_freeze_report_v2.2.md` | 2025 核心期、2026 provisional 与 TA/DIC 推理覆盖更正 | 当前 P1 的正式数据入口 |
| `p0_freeze_report_v2.1.md` | 首次冻结记录 | 已由 v2.2 替代；仅保留审计 |
| `backup/model_design_review_20260906.md` | ST 模型容量、优化和训练审计 | 历史诊断 |
| `backup/controlled_experiments_20260906.md` | fCO2 受控容量/优化实验 | 有效开发实验附件 |
| `backup/chla_ablation_20260906.md` | MODIS Chl-a 消融 | 有效开发消融附件 |
| `backup/ta_dic_constraint_review.md` | TA/DIC 稀疏监督和约束问题 | 结构决策依据 |
| `backup/carbonate_identifiability_redesign_20260909.md` | 碳系统自由度和结构派生 DIC | 当前结构设计依据 |
| `backup/weighted_carbonate_results.md` | 支持度加权实验 | 路线诊断；不能覆盖新 CODAP 结论 |
| `backup/codap_na_v2026_download_audit_20260928.md` | CODAP 下载、QC 和覆盖 | 数据审计附件 |
| `backup/loss_design.md` | 旧联合损失方案 | 历史设计；与当前结构方案冲突时以后者为准 |
| `backup/design.md` | v2 初始软件/模型设计 | 工程背景，不代表当前实验口径 |
| `backup/migration_v1_to_v2.md` | v1/v1.1 迁移 | 历史兼容说明 |

## 维护规则

- 当前数值只在总账出现一次；详细报告链接到总账，不复制成新的“最新结论”。
- 新实验在训练前复制 `../configs/experiment_record_template.yaml`，完成后登记到实验总账。
- 大型数据、checkpoint 和逐行预测不进入 Git；正式实验的汇总表、reviewer-ready 图片、报告与归档 manifest 必须进入 `experiment_archive/`，并引用大型产物哈希。
- 错误实验不删除，标记为 X；测试集一旦用于选择，立即在总账降级为 benchmark。

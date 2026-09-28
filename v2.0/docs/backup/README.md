# 历史记录备份

本目录保存已被统一总账吸收的旧设计、实验报告和数据审计。文件保留原始表述以便追溯，不代表当前产品定义或验证状态。

当前信息从以下文件读取：

- `../data_registry.md`：数据版本、路径、哈希、QC 和限制；
- `../experiment_registry.md`：实验方法、关键指标、证据等级和失效状态；
- `../decision_log.md`：当前决策、替代关系和待决策事项；
- `../current_status_and_roadmap_20260928.md`：P0–P3 路线和通过门槛；
- `../experiment_protocol.md`：共同验证规范。

## 文件分类

| 文件 | 分类 | 使用方式 |
|---|---|---|
| `design.md` | 初始模型/软件设计 | 理解代码最初意图；当前科学设计以总账为准 |
| `loss_design.md` | 旧联合损失设计 | 解释现有代码；不自动代表下一轮正式损失 |
| `migration_v1_to_v2.md` | 迁移记录 | 核对 v1/v1.1 兼容性 |
| `model_design_review_20260906.md` | 模型诊断 | 容量、优化和训练问题证据 |
| `controlled_experiments_20260906.md` | 实验报告 | fCO2 受控开发实验明细 |
| `chla_ablation_20260906.md` | 实验报告 | Chl-a 消融明细；旧 benchmark 已打开 |
| `ta_dic_constraint_review.md` | 约束审计 | TA/DIC 稀疏监督问题依据 |
| `carbonate_identifiability_redesign_20260909.md` | 结构分析 | DIC 结构派生方案依据 |
| `weighted_carbonate_results.md` | 实验报告 | 支持度加权路线明细 |
| `codap_na_v2026_download_audit_20260928.md` | 数据审计 | CODAP 下载和初步 QC 明细 |

如果旧文件与根目录总账冲突，不修改旧文件来伪装历史一致；在 `../decision_log.md` 记录替代关系。

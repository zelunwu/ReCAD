# 2026-09-06 受控训练实验

目的：区分 fCO2 单任务模型的优化、局地解码和主干容量限制。协议及源码哈希在首次训练前写入 `outputs/experiments/controlled_20260906/protocol.json`。

本轮只使用原 training/dev 集合，所有 checkpoint 选择在完整 38,949 条 dev 标签上进行。没有打开时间测试或空间挑战标签评分；这些曾被查看的集合不能成为本轮新的盲评证据。多变量化学反演和新外部独立集不在此次容量诊断范围。

## 固定条件

- 数据、统计量与 split 哈希写入 data_manifest.json；target 与 predictor 统计均只使用 training。
- 沿用实际 patch8、1,077 个 token 和原背景聚合方法、原观测足迹 mask。配置明确校验 patch8，不改变训练区域来混淆模型比较。独立地理 mask 和逐通道 token coverage 作为后续数据设计实验。
- 只解码实际监督查询，所有环境 token 仍可参与上下文。其值和梯度与原 dense decoder 的等价性由单元测试覆盖。
- attention 使用 SDPA 减少显存中间量；无 dropout 时与原 attention 输出等价，含全空 mask 的反向传播为有限值。
- 单成员 seed100，无年份 bootstrap。已修复通用 dataset 的 bootstrap 漏排年份问题，新增回归测试。
- 每步先按该年份 training 标签数比例抽取年份，再在年份内等概率有放回抽取 512 条标签；每条 training 标签有相同边际采样概率。只用 training 标签计数设计采样，保留年份批次的上下文一致性。
- AdamW，基础 weight decay=1e-4，gradient clipping=1，BF16，前 100 steps warmup。0–2000 steps 使用基础 LR；2000–5000 使用 0.3×LR；后续使用 0.1×LR。
- 每 250 steps 对全部 train/dev 标签做 eval-mode 评分，记录真 R2、RMSE、MAE、bias、coverage、采样 loss、gradient norm、LR、时间；保存 best 与完整可续训 last 状态及 RNG、标签曝光计数。

## 条件

| ID | 结构/损失 | LR | 预算 |
|---|---|---:|---:|
| tiny_st96 | 原 D96，MSE，无 dropout/weight decay | 3e-4 | 2,000 steps |
| tiny_deep_decoder | D96 + 深残差解码器，同一 tiny 集 | 3e-4 | 2,000 steps |
| st96_mse_lr1e4 | 原 D96，MSE | 1e-4 | 2,000 steps |
| st96_mse_lr3e4 | 原 D96，MSE | 3e-4 | 2,000 steps |
| st96_mse_lr1e3 | 原 D96，MSE | 1e-3 | 2,000 steps |
| st96_gaussian | 原 D96，Gaussian NLL | 3e-4 | 2,000 steps |
| local_resnet256 | 256 宽、3 个残差块的单点模型，MSE | 3e-4 | 2,000 steps |
| st96_deep_decoder | 原 D96 主干 + 256 宽、3 残差块解码器，MSE | 3e-4 | 2,000 steps |
| st192_mse | 3+3 层、D192，MSE | 3e-4 | 2,000 steps |
| st256_mse | 3+3 层、D256，MSE | 3e-4 | 2,000 steps |

Tiny 集为 seed719 在 training 标签最多的年份中抽取 2,048 条标签，固定全年背景。已核对 2,048 条 `(month, patch, cell features)` 输入互异，无相同输入对应冲突标签的阻碍。其拟合误差仅用于优化诊断，不表示泛化能力。

阶段结束后按照预先登记规则，选择最佳 dev 的 ST96 MSE 学习率，以及 local/deep-decoder/D192/D256 中最佳 dev 的结构，分别继续到 5,000 steps。较早保存的 best checkpoint 继续保留；不以预算结束点替代最佳 dev 选择。

容量、损失、局地头比较首先使用同一 LR=3e-4、相同 2,000 steps 条件；主干 LR 搜索和最终延长单独解释。各结构只有一个随机种子，扩容结果仅为开发探索，不给显著性或外部泛化结论。2,000 steps 约 6.48 次/标签、5,000 steps 约 16.19 次/标签平均曝光，不称 2,000/5,000 个完整 epoch。

复现命令（从 v2.0 执行）：

```powershell
& .venv/Scripts/python.exe -u scripts/run_st_controlled_experiments.py --run suite
```

可通过 `--run <ID> --steps <budget>` 执行/恢复单条件，`--run report` 刷新汇总图。协议与源码发生变化时 runner 拒绝在原目录继续，须另建版本，避免静默覆盖实验定义。

## 已完成结果

10 个条件与两次延长训练全部正常完成，总计 26,000 次 optimizer updates。所有正式条件对完整 training/dev 评分，预测覆盖率 100%；保存的预测成员集合和 RMSE 已逐一重算核对。23 项相关单元/集成测试通过（模型、tensorize、bootstrap、sparse/dense 预测和梯度等价、SDPA 空 mask、ensemble）。

相同 2,000 steps、相同 LR=3e-4 的主要对照：

| 模型 | 参数量 | Train RMSE | Dev RMSE | Dev 真 R2 |
|---|---:|---:|---:|---:|
| D96，MSE | 690,433 | 23.845 | 28.386 | 0.626 |
| D96，Gaussian NLL | 690,530 | 27.231 | 28.831 | 0.614 |
| D96 + 深解码器 | 1,117,761 | 24.723 | 27.799 | 0.641 |
| D192，MSE | 2,744,833 | 22.836 | 27.810 | 0.641 |
| D256，MSE | 4,872,193 | 24.330 | 29.000 | 0.609 |
| 单点 ResNet256 | 400,385 | 29.953 | 30.468 | 0.569 |

全部 RMSE 单位 µatm。表中 train 和 dev 来自同一最佳开发检查点；单点 ResNet 的最佳点为 1,750 steps。原 D96 的三档 LR=1e-4/3e-4/1e-3，在 2,000 steps 预算下的最佳 dev RMSE 为 29.198 / 28.386 / 28.088。

固定 tiny 集最佳拟合 RMSE：原 D96 为 8.687（2,000 steps），深解码器为 6.733（1,750 steps）。不用于推断泛化。

延长到 5,000 steps 后：

| 条件 | 选中步骤 | Train RMSE | Train 真 R2 | Dev RMSE | Dev 真 R2 |
|---|---:|---:|---:|---:|---:|
| 原 D96，MSE，初始 LR=1e-3 | 2,750 | 20.736 | 0.833 | 27.326 | 0.653 |
| D96 + 深解码器，初始 LR=3e-4 | 5,000 | 19.782 | 0.848 | 27.643 | 0.645 |

原 D96 的最后一步 train RMSE=18.595，dev RMSE=28.123；因为 dev 更差，保留 2,750 步权重。该结果表明训练拟合可继续改善，但开发收益趋缓。两条件训练结束时平均每条标签曝光 16.189 次，未使用过的 training 标签为 0；这些计数描述结束点，不代表更早的最佳 checkpoint。

证据判断：

- 现有 D96 的训练拟合不是锁死在原来的 26.46；损失和优化设置确实值得调整。
- 同一预算下，MSE 相比 NLL 的开发 RMSE 改善仅约 1.5%；更明显的 train 改善不能等同于泛化改善。
- 深解码器与 D192 在 2,000 steps 下几乎持平（dev 差 0.011），不能认定一个优于另一个。
- D256 在本次学习率/预算下更差，扩容没有单调收益；它未获得完整独立调参，不能推导所有大模型无效。
- 深解码器 tiny 优势没有转化为相对已优化 D96 的明确 full-dev 优势；本轮不支持单凭 tiny 分数替换主模型。
- 单点残差网络在本次输入编码、采样、学习率下较差，不意味着所有单点模型都不合适。与旧 MLP 的归一化/预算等条件不完全相同。
- 旧五成员 Gaussian 集成的 train/dev=26.457/27.946，新单成员最佳为 20.736/27.326，仅作历史参考：训练改善约 21.6%，dev 改善约 2.2%。多个训练条件同时变化，且成员数不同，不能将此差异归因于一个因素。

本轮仍为单 seed 的开发探索，未进行新的 test/外部独立评分。优化和局地结构有收益，但尚无“泛化瓶颈已经解决”的证据。下一阶段应固定候选、做多种子复现及同数据树模型基线，再以审计后的独立观测确认。

产物位于 `outputs/experiments/controlled_20260906/`：`model_summary.csv` 为各条件当前最佳指标（预算列明确），`screen_2000_summary.csv` 为固定预算对照，`learning_curves.png` 为诊断曲线。各条件 `screen_2000/` 保留延长前权重/历史/预测；`best.pt`、`last.pt` 分别用于所选模型与续训。完整协议和数据/源码哈希均已保存。

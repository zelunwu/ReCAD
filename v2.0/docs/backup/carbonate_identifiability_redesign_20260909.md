# 稀疏 TA/DIC 条件下的碳酸盐系统重构设计

## 问题定义

在给定温度和盐度后，海水碳酸盐系统仍有两个独立自由度。当前网络分别输出 fCO2、TA 和 DIC，再用
`CO2SYS(T, S, TA, DIC) ≈ fCO2` 作为软损失。对没有 TA/DIC 标签的大多数网格，这只把三个预测限制在一个允许曲面上，不能确定曲面上的位置。稀疏标签、Carter TAest 偏差与网络正则共同决定最终位置，因此可能得到闭合但不真实的 TA/DIC。

## 推荐参数化

网络保留 SSS 和 fCO2 预测，只再预测一个带不确定性的 TA 状态：

`TA = TA_mixing(S, region, month) + delta_TA_nonconservative(x)`

其中 `TA_mixing` 是按海区或流域部分汇聚的咸水/淡水端元混合先验，`delta_TA_nonconservative` 使用 SST、Chl-a、距岸、地形、季节，以及可获得时的径流、氧和营养盐。模型同时输出 `mu_delta` 和 `sigma_delta`。DIC 不设独立自由输出头，而由可微 CO2SYS 解码器计算：

`DIC_hat = CO2SYS_inverse(T, S_hat, fCO2_hat, TA_hat)`

因此输出仍包含 SSS、fCO2、TA 和 DIC，但化学上只有两个独立碳参数，闭合由结构保证。

## 监督方式

- SOCAT 点：监督 fCO2；TA/DIC 保持为先验分布，不把 TAest 当真值。
- GLODAP 点：观测 TA 直接监督 TA 分布；观测 DIC 通过派生的 DIC_hat 反向监督同一个 TA 状态。
- 同时有 TA、DIC 的点可用 exact CO2SYS 计算 fCO2，作为带有传播不确定性的辅助标签。
- Carter TAest 作为低保真先验均值或额外特征，并学习 region/month-dependent bias；其损失权重由已测误差决定。
- 使用 Student-t 或异方差 Gaussian NLL，避免少数航次异常值主导 MSE。
- 对无碳参数观测的网格必须输出较宽区间；不能把化学闭合误当作低不确定性。

## 现有数据的数值审计

`scripts/audit_carbonate_parameterization.py` 使用现有 GLODAP train/dev/2004–2005 状态和 exact PyCO2SYS 比较两种逆向参数化。结果在
`outputs/experiments/carbonate_parameterization_20260909/sensitivity.csv`。

- 固定 fCO2 时，`dDIC/dTA` 的中位数为 0.83–0.87。
- 固定 fCO2 时，`dTA/dDIC` 的中位数为 1.15–1.20。
- 假设 fCO2 误差标准差 20 µatm、TA 或 DIC 误差标准差 40 µmol/kg，`fCO2+TA -> DIC` 的派生 RMSE 为 33–40 µmol/kg，`fCO2+DIC -> TA` 为 49–53 µmol/kg。

所以当前区域优先选择 `fCO2 + TA -> DIC`，但最终决定仍应使用 expocode 五折 CV 和真正未打开的外部验证。

## 下一组受控实验

1. C0：旧 B2，作为冻结基线。
2. C1：确定性 TA 头，DIC 由 CO2SYS 解码；验证结构闭合本身。
3. C2：异方差 TA 头，训练观测似然并报告覆盖率；验证不确定性是否校准。
4. C3：C2 加海区/季节混合端元先验；验证近岸 TA 与跨航次表现。
5. C4：若径流数据就绪，在 C3 增加径流和距河口信息；只在 C3 胜出后进行。

模型选择同时看四目标 RMSE、R2、expocode 五折方差、exact CO2SYS 误差，以及 TA/DIC 50%/90% 预测区间覆盖率。2004–2005 已打开，只作 benchmark；最终结论需要新的外部盲测集。

## 边界

这个设计消除模型内部多余的碳参数自由度，并能诚实表示信息不足，但不能从单个 fCO2 观测唯一恢复 TA 和 DIC。真正缩小后验仍需要第二种独立碳参数观测，优先考虑 pH、TA 或 DIC 的沿岸浮标、BGC-Argo、区域调查与 CODAP 数据。

# ReCAD 当前状态与下一阶段实验路线（2026-09-28）

> **历史快照 / 已被替代（2026-10-08）**：本文保留 2026-09-28 时点的推理与数据盘点，不再作为 P2 执行权威。当前产品范围、证据状态与标签暴露以 `configs/frozen/product_scope_v2.3.json`、`docs/experiment_archive/p2_scope_reconciliation_v2.3/REPORT.md` 及 GitHub #1/#4/#5 为准：SSS/fCO2 是全球沿海核心研究目标；SSS 当前 `diagnostic_only` 且旧 locked set 已用尽；fCO2 仅 LME 12 `pass_regional`；TA/DIC 是北美非阻断可选扩展。

GitHub 执行层级：父 Roadmap [#1](https://github.com/zelunwu/ReCAD/issues/1)；P0 数据与验证冻结 [#2](https://github.com/zelunwu/ReCAD/issues/2)；P1 单任务产品可行性 [#3](https://github.com/zelunwu/ReCAD/issues/3)；P2 空间自适应与结构化碳系统 [#4](https://github.com/zelunwu/ReCAD/issues/4)；P3 冻结候选与盲测 [#5](https://github.com/zelunwu/ReCAD/issues/5)。#2–#5 已通过 GitHub sub-issue 关系挂到 #1。

## 1. 产品范围决定

ReCAD 下一阶段采用两层产品，而不再要求一个全球模型同时可靠输出四个变量。

| 层级 | 空间范围 | 输出 | 定位 |
|---|---|---|---|
| 全球沿海主产品 | 全球 1/8° 沿海 | SSS、fCO2 | SSS 是对 GLORYS 背景的沿岸偏差订正；fCO2 是主要反演目标 |
| 北美沿岸碳酸盐扩展 | 美国东西岸，后续可扩展至加拿大、墨西哥、阿拉斯加 | TA、由 CO2SYS 派生的 DIC，并同时保留 SSS/fCO2 | 只在观测支持和适用域通过检查的区域发布 |

DIC 不设置独立自由输出头。在 TA 适用域内使用
`DIC = CO2SYS_inverse(T, SSS, TA, fCO2)`，并传播 SSS、fCO2、TA 的预测不确定性。
全球无 TA 支持区域不发布 DIC 产品。

## 2. 当前数据状态

### 2.1 已落地数据

| 数据/产物 | 当前状态 |
|---|---|
| 全球月场 | 1993–2026，408 个月，1297×2880；148,795 个 coastal-mask 格点 |
| 全球 prepared | `outputs/prepared_global_p32.nc`，约 39.75 GB；SST、GLORYS SSS、ADT、wspd、pCO2air、SOCAT fCO2 |
| 全球 SOCAT 联合缓存 | 752,668 train / 239,396 validation / 48,934 原 2004–2005 benchmark 格月 |
| 全球现场盐度 | 19.0 M 条 SOCAT 盐度聚合为 890,581 个格月；有效 train/validation/benchmark 为 628,080 / 220,295 / 40,844 |
| GLODAP 表层 TA/DIC | 14,930 个深度≤5 m 锚，592 个航次；全球缓存聚合后 10,359 / 2,824 / 784 |
| 美国大西洋区域 | 408 月，441×480；有 MODIS Chl-a、SOCAT 盐度和完整物理输入 |
| 美国太平洋区域 | 408 月，321×200；有 MODIS Chl-a、SOCAT 盐度和完整物理输入 |
| MODIS Chl-a | 区域缓存覆盖 2002–2020；尚无全球标准缓存 |
| 可微 CO2SYS | 正向和逆向代理均已完成；逆 DIC 代理宽域 RMSE 0.552 µmol/kg |

原始输入约 775 GB：CCMP 387 GB、SSS 202 GB、海平面 92 GB、SOCAT 74 GB、SST 19 GB、GLODAP/OCADS 约 1 GB。

### 2.2 数据问题

1. 2004–2005 已被多轮用于比较、调参与模型选择，只能称为历史 benchmark，不能称为 independent test。
2. 2026 年在当前 SOCAT 缓存中仅有 36 个格月记录（train 26、validation 10），属于不完整年份；正式数据版本应先截止到 2025，或截止到 SOCAT 清单确认的最后完整年份。
3. 全球 Chl-a 尚未构建，因此目前全球模型是 no-Chl-a baseline。
4. 现有 SSS 结果是用 GLORYS SSS 作背景、SOCAT 现场盐度作监督得到的偏差订正结果，不是完全不依赖再分析场的独立反演。
5. GLODAP 对北美陆架覆盖不足。全球有 TA/DIC 锚点的沿海格只有 1.4%，美国东西岸也只有约 3.3–3.5%。

### 2.3 新的数据机会：CODAP-NA V2026

NOAA/NCEI 已发布 CODAP-NA Version 2026（NCEI Accession 0315529），时间覆盖 1981-08-23 至 2024-11-23，包含 446 个航次、32,250 个剖面和 14 个水文、生物地球化学变量，包括 TA、DIC、pH、fCO2、氧及营养盐。它覆盖北美全部陆架，是下一阶段 TA/DIC 的首选数据源。

正式使用前必须按 EXPOCODE、时间、经纬度和瓶号与 GLODAP 去重，并记录每条观测的来源谱系。CODAP 不能在去重前直接作为“外部独立验证”，因为其中部分航次可能同时存在于 GLODAP 或其他综合产品。

- 数据页：https://www.ncei.noaa.gov/access/ocean-carbon-acidification-data-system/synthesis/CODAP-NAv2.html
- 数据说明：https://catalog.data.gov/dataset/coastal-ocean-data-analysis-product-in-north-america-codap-na-version-2026-from-1981-08-23

## 3. 已完成实验及可信结论

| 实验 | 状态 | 可用结论 |
|---|---|---|
| 受控 fCO2 模型实验 | 完成 | ST 模型能学习区域 fCO2；单纯增大主干不是主要瓶颈 |
| MODIS Chl-a 消融 | 完成 | 在 2002–2020 共同样本上明显改善 fCO2，但旧 2004–2005 结果已被打开，只能作 benchmark |
| 四目标联合模型 B1/B2 | 完成 | CO2SYS 软损失把闭合误差降低约 92%，预测技能只改善约 0–3%；约束不会创造 TA/DIC 信息 |
| C1–C3 结构化碳酸盐模型 | 完成 | 独立预测 SSS/fCO2/TA、结构派生 DIC 可实现小于 0.6 µatm 的 exact 闭合 |
| 美国东西岸实验 | 完成 | 东岸 TA 留航次 RMSE 约 38–51；西岸约 75–127，随机点 test 明显乐观 |
| 空间支持度加权化学约束 | 完成 | 减少无观测区伪约束；改善有锚区 validation，但没有解决跨航次外推 |
| 全球 4° patch weighted 模型 | 完成 | SSS 可用；fCO2 validation RMSE 60.45、旧 benchmark RMSE 53.20 且 skill −0.138，全球统一模型未过关 |
| 最新 checkpoint-selection smoke | 失败 | `run_weighted_carbonate.py` 调用了未定义的 `selection_score`；下一轮训练前必须修复并加测试 |

全球 SSS 的旧 benchmark RMSE 为 1.08 PSU、R² 0.986，但高 R²受到波罗的海和河口极端盐度扩大目标方差的影响。美国窗口 RMSE 1.17–1.45 PSU、相对均值基线 skill 约 0.39–0.46。结论应表述为“全球尺度和大盐度梯度上稳定，精细近岸订正仍需分区验证”，不能笼统写成所有沿岸都达到极高精度。

## 4. fCO2 的新验证方案

### 4.1 放弃单一年份 test，评价三种泛化问题

同一个 test 无法同时回答插值、空间迁移和时间外推。下一轮建立三个相互独立的外层评价任务：

1. **新航次插值**：按 EXPOCODE 整组留出，同一航次不能跨 train/validation/test；衡量已观测海区对新航次的重建。
2. **新区域迁移**：按 LME/沿岸省或连续空间块整区留出；衡量无邻近观测海区的外推。
3. **未来时间外推**：采用 forward chaining，例如 train≤2018、validation=2019–2021、locked test=2022–2024；不让未来年份进入输入统计量或 checkpoint 选择。

开发阶段用嵌套 grouped CV：内层只选模型和超参数，外层只估计泛化。2004–2005 保留为 v1.1 可比性附表，不参与任何选择。

### 4.2 真正外部数据

最终 fCO2 外部验证优先使用固定站/平台，而不是再从同一 SOCAT 表随机抽点：

- OCADS 全球 CO2 Time-series and Moorings 数据；
- OceanSITES pCO2 时间序列；
- OCADS/PMEL Saildrone 或其他自主平台任务；
- 若论文时间允许，使用冻结模型之后发布的新 SOCAT 数据版本做 prospective validation。

OCADS 的部分数据会提交给 SOCAT，因此“不同下载入口”不等于独立。必须以平台、EXPOCODE、时间和空间近邻对 SOCAT v2026 做交叉匹配，只有未进入训练谱系的数据才能进入 external test。外部集在模型、变量、超参数和报告规则冻结后一次性打开。

### 4.3 评分单位

同时报告三种权重，避免高频航次主导结果：

- grid-month 等权：对应月产品用途；
- cruise 等权：每个航次先计算误差，再对航次平均；
- region 等权：每个 LME/沿岸省同权。

每种权重报告 RMSE、MAE、bias、R²、相对气候态/区域季节气候态的 skill，以及按航次 block bootstrap 的 95% 区间。逐观测 RMSE只作仪器/代表性误差附表；当前全球同格月内 fCO2 pooled std 约 26.7 µatm，因此产品主指标应使用 grid-month 聚合。

## 5. 下一组模型实验

### 5.1 全球 fCO2/SSS 主产品实验 F0–F5

所有实验使用同一冻结数据和三类 grouped outer folds。

| 条件 | 改动 | 回答的问题 |
|---|---|---|
| F0 | 区域×月份气候态、线性模型、GBDT/MLP 点模型 | 模型至少要超过哪些简单基线 |
| F1 | fCO2-only ST 模型 | 移除 SSS/TA 任务竞争后，fCO2 是否恢复 |
| F2 | SSS 与 fCO2 共享背景编码，但使用独立 adapter/head；与完全分离模型对照 | 共享信息究竟帮助还是损害 fCO2 |
| F3 | 全球共享主干 + LME/水团 mixture-of-experts，区域 adapter 部分汇聚 | 单一全球函数是否应改为区域函数族 |
| F4 | 1°局地沿岸上下文 + 4°大尺度上下文的多尺度模型 | 保留河口、上升流和陆架锋面，同时控制全球 attention 计算量 |
| F5 | F4 增加 Chl-a、混合层深度、海深/坡度、距岸/河口、上升流和径流变量 | 误差是否来自缺少过程变量 |

F1 是下一轮第一优先级。当前全球模型把 SSS 与 fCO2 放在共享主干中，而 SSS 信号远强于 fCO2；先证明单任务 fCO2 能否提高，再决定是否值得做 MoE 和多尺度结构。

SSS 单独按残差订正产品训练和评价，至少比较 `GLORYS 原场`、`线性区域订正`、`ST 残差订正`。按盐度范围、距岸、河口/非河口、LME 分层报告，避免全球 R²掩盖局地问题。

### 5.2 美国沿岸 TA 实验 A0–A4

先构建去重后的 CODAP-NA V2026 + GLODAP 表层数据集，再按海岸、子区、季节和航次分层。推荐的 TA 结构是：

`TA = alpha(region, season) + beta(region, season) * SSS + delta_bio(x)`

`alpha` 和 `beta` 使用层级部分汇聚：数据多的区域由观测决定，数据少的区域向海岸级均值收缩。Carter/ESPER 只作为先验均值或输入，不作为标签。

| 条件 | 模型 |
|---|---|
| A0 | 每个区域固定 TA–SSS 线性回归 |
| A1 | Carter/ESPER 原值及 region-season bias correction |
| A2 | 层级 varying-coefficient TA–SSS 模型 |
| A3 | A2 + SST、Chl-a、月份、海深、距岸/河口的非线性残差 |
| A4 | A3 + 氧、营养盐、MLD、径流；只在 A3 确认有剩余可预测信号后运行 |

评价只使用留航次和留子区外层折。除总体误差外，必须报告每个子区的斜率、截距、残差尺度、航次数和 OOD 标志。生物过程强、TA–SSS 线性失效的区域允许输出更宽区间或不发布，而不是强制一个错误的线性关系。

### 5.3 DIC 与 CO2SYS 实验 C0–C3

| 条件 | 方法 |
|---|---|
| C0 | 无化学项；只检查观测 TA/DIC 的经验基线 |
| C1 | 结构派生 DIC：`fCO2 + TA -> DIC`，不另加 closure loss |
| C2 | C1 + 配对 DIC 观测似然；DIC 误差反向更新 TA 残差模型 |
| C3 | C2 + 支持度加权的弱辅助化学项，lambda 做 0/0.01/0.05 消融 |

因为 C1 已结构性闭合，额外 CO2SYS penalty 不是必需项。它只在观测支持区作为辅助正则，不能施加到全球无 TA 信息区。最终用 Monte Carlo 或 delta method 将 fCO2、SSS、TA 的联合不确定性传播到 DIC，并报告区间覆盖率。

## 6. 执行顺序与停止条件

### P0：已于 2026-09-28 完成并冻结

1. [x] 修复 `selection_score` 缺失，并用回归测试约束密集/稀疏两类任务的 checkpoint 选择。
2. [x] 冻结 `data_manifest_v2.2.json`：数据轴 1993–2026；SSS/fCO2 核心期到 2025；TA/DIC 观测到 2024、预测可到 2026；含 QC、availability、空间 metadata 和哈希。
3. [x] 2004–2005 统一降为历史 benchmark，不再用于正式模型选择。
4. [x] CODAP-NA V2026 与 GLODAP 已合并、标记重复组和 primary 记录。
5. [x] 冻结航次 grouped split、五折 CV、leave-LME-out 和 forward-chain；41 个新 CODAP 航次封存为碳参数外部集，未来 SOCAT 增量封存为 SSS/fCO2 外部集。

验收证据和复现命令见 `docs/p0_freeze_report_v2.2.md`。P1 必须使用 v2.2 manifest；v2.1 已被替代。任何数据或 split 改动需继续提升版本。

### 空间非平稳性：全局主干 + 软门控区域专家

后续方法检索固定为两条并行路线：一条跟踪沿岸碳循环、动态生物地球化学省、观测代表性和区域产品；另一条跟踪计算机科学的 spatial mixture-of-experts、group distributionally robust optimization、连续空间参数化、图神经网络和概率校准。不能因为地球科学论文仍使用较旧的回归骨干就停止检查新算法，也不能因为计算机模型更新就忽略碳酸盐可识别性和海洋过程。

经纬度编码不足以处理沿岸非平稳性。统一模型会被北大西洋、美国沿岸等高密度观测区支配；完全按 N 个区域独立训练又会在边界产生跳变，并让稀疏区失去跨区借力。主候选改为：

`y_hat(x) = y_global(x) + sum_k gate_k(z) * delta_k(x)`，其中 `gate_k >= 0` 且 `sum_k gate_k = 1`。

- `y_global` 学习全球共同关系；区域 expert 只学习有收缩约束的残差，低数据区自然退回全局模型。
- gate 使用 SST/SSS 状态、Chl-a、海冰、混合层、地形、距岸、河流/上升流代理、季节和海洋连通性；lon/lat 只作位置编码，不能单独决定专家。
- 使用 top-2 软路由和重叠专家，不生成硬省界。gate 加空间/时间平滑与负载均衡，但在锋面、河口和陆地阻隔处降低平滑权重。
- SSS、fCO2、TA 可以共享环境编码器，但允许不同的 task gate；DIC 仍由 CO2SYS 结构派生。
- 训练 batch 按 region → cruise → month 分层采样，损失同时报告/优化区域宏平均与 worst-group/CVaR，不能按观测点 pooled MSE 决定模型。
- 邻域关系优先采用仅沿水体连通的 coastal graph 或 coast-aware attention，避免海峡两岸、半岛两侧和岛屿附近仅因球面距离近而错误交换信息。

P1/P2 增加空间结构对照：统一模型、统一模型+区域平衡、全局主干+区域残差 adapter、环境软门控 experts、软门控+沿岸图。硬分区独立模型保留为地球科学基线，不作为默认产品。除常规 RMSE 外，报告 macro-LME、worst-LME、整 LME 留出、边界带跳变、观测支持距离和区间覆盖率。只有软门控方案在这些指标上胜过统一模型与硬分区，才进入正式结构。

### P1：先回答产品能否成立

1. 跑 F0、F1、F2；若 fCO2-only 仍不能稳定超过区域季节气候态和 GBDT，则暂停深模型，优先补变量/分区。
2. 跑 SSS 三基线并完成按河口、距离和盐度段的误差审计。
3. 用 CODAP/GLODAP 跑 A0–A2；只有 A2 在留航次折稳定超过 Carter 和分区线性模型，才继续 A3/A4。

### P2：结构升级

1. fCO2 通过 P1 后再跑 F3/F4；Chl-a 和过程变量放在 F5 单独消融。
2. TA 通过 A2/A3 后再跑 C1–C3 并发布 DIC。

### 产品门槛

- fCO2：在新航次、新区域、未来时间三类外层评价中均优于预注册基线，且 external platform test 不出现系统偏差。
- SSS：在 GLORYS 原场之上取得稳定 skill，并在河口/近岸分层中不出现大面积退化。
- TA：留航次和留子区均优于 Carter 与区域线性基线，90% 区间覆盖经过航次级校准。
- DIC：只在 TA 门槛通过的格点发布；除 RMSE 外要求 exact CO2SYS 闭合和区间覆盖均通过。

## 7. 当前最合理的论文叙事

1. 全球沿海 SSS/fCO2 是主任务，但 SSS 和 fCO2 分别建模或使用弱共享结构。
2. SSS 是具有现场观测监督的再分析偏差订正产品；fCO2 是真正的稀疏观测时空重建产品。
3. 北美沿岸 TA 是区域扩展：用可解释的区域 TA–SSS 混合关系和生物过程残差建模。
4. DIC 是在 TA 适用域内由 CO2SYS 派生并传播不确定性的诊断产品。
5. 2004–2005 只用于与 v1.1 的历史可比性；主要可信度来自 grouped nested CV 和锁定的外部平台验证。

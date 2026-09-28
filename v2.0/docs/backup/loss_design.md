# 联合反演损失函数设计(ReCAD v2.0+)

> 本文档定义"一个模型同时反演 SSS / TA / fCO2 / DIC"的损失函数设计。
> 所有结论基于 v2.0 数据栈(SOCAT 散点 + GLODAP 锚点 + CO2SYS 化学闭合),
> 与 `docs/backup/design.md`(ST-Transformer)、`docs/data_sources.md` 配套。
> 初稿 2026-09,配合 `scripts/run_mvp.py` / `recad.train` 迭代。

---

## 1. 目标与约束

联合反演模型使用同一套预测因子(**SST, SSS(再分析), ADT/SSH, pCO2air, u10**,
+ 经纬度/月份),**同时输出**:

| 输出 | 单位 | 化学角色 |
|------|------|---------|
| fCO2̂ | µatm | 主任务(海表 CO2 逸度) |
| SSŜ | PSU | 反演链锚(盐度驱动 TA) |
| TÂ | µmol/kg | 碳状态量(与 SSS 强相关) |
| DIĈ | µmol/kg | 碳状态量(化学闭合) |

**化学闭合**:给定 (T, S, TA, DIC),CO2SYS 确定性计算 fCO2——
`fCO2 = CO2SYS(T, S, TA, DIC)`。模型输出必须与 SOCAT 观测在化学上自洽。

**SST 不进输出**:已作为预测因子输入,精度足够(OISST),不设 SST 输出头。

---

## 2. 多路独立监督(核心原则)

**样本不需要同时拥有所有标签。** 每条观测只贡献它持有的那部分损失,
训练规模由 SOCAT 主导(千万级),GLODAP 稀疏锚只钉化学绝对刻度:

```
L_total = Σ_i λ_i · mean( (residual_i / σ_i)² )
```

| 项 | 监督来源 | 样本量级 | 物理角色 |
|----|---------|---------|---------|
| L_fCO2 | SOCAT fCO2 观测 | ~10⁷ | 主任务 |
| L_SSS | SOCAT 现场 sal | ~2×10⁷(93% 覆盖) | 盐度→TA 链的独立锚 |
| L_TA | GLODAP 表层 TA | ~10³–10⁴ | 碳化学绝对刻度校准 |
| L_DIC | GLODAP 表层 DIC | ~10³–10⁴ | 同 TA |
| L_chem | 全部 SOCAT 点(化学闭合) | ~10⁷ | 化学把稀疏监督变稠密 |
| L_smooth | 空间网格输出 | — | 正则(见 §5) |

> **为什么数据量不降**:L_fCO2/L_SSS/L_chem 都用 SOCAT 全量(千万级);
> GLODAP 只在 L_TA/L_DIC 稀疏出现,是"锚"不是"训练主体"。

---

## 3. 单位统一:噪声归一化(χ² 化)

不同变量的单位(µatm / PSU / µmol/kg)不可直接加权,标准做法是**除以观测误差**:

```
L_i = mean( ( residual_i / σ_i )² )      （无量纲,偏差相对观测噪声的倍数）
```

### 3.1 观测误差建议初值

| 项 | 单位 | σ 初值 | 依据 |
|----|------|--------|------|
| L_fCO2 | µatm | 5.0 | SOCAT 推荐值精度 ~2–5 µatm |
| L_SSS | PSU | 0.1 | 船测盐度精度 0.01–0.1 |
| L_TA | µmol/kg | 8.0 | GLODAP 精度 2–8 |
| L_DIC | µmol/kg | 8.0 | GLODAP 精度 2–8 |
| L_chem | µatm | 5.0 | 与 fCO2 同量级 |

### 3.2 进阶:逐点 σ

fCO2 的逐点 σ 可用 `socat/binned/fco2_binned.nc` 的 **`std_err`**(格内标准误):
观测密集处要求更严,稀疏处放松 —— 最优加权(与 v1.1 的输入误差
`recad.constants.U_INPUTS` 思路一致,但这里作用于损失而非传播)。

---

## 4. λ 权重设计

归一化后各 loss 量级相当,**λ 初始全 1**,再按原则微调:

| 原则 | 做法 |
|------|------|
| 主任务优先 | λ_fCO2 = 1.0(基准参照) |
| 稀缺锚补权 | λ_TA = λ_DIC = **5–20**(GLODAP 样本少,否则被淹没) |
| 化学自洽 | λ_chem = 1.0(与主任务同量级) |
| 平滑正则 | λ_smooth = 0.01–0.1(只防噪,不主导) |
| 盐度锚 | λ_SSS = 1.0 |

### 4.1 数据量悬殊的三重补偿

1. **任务内 mean**:每个 L_i 独立在各自样本上平均,不混样本数;
2. **GLODAP 锚点过采样**:训练 batch 内混入固定比例锚点(如每 batch 保底
   一定数量),弥补"出现频率低";
3. **λ 补偿**:上述 5–20,与过采样配合。

### 4.2 λ 灵敏度扫描(论文方法学)

每个 λ ∈ {0.1, 1, 10}(或 3 点网格)跑小规模训练,监控测试集 R²/RMSE,
选稳定区间。典型做法见 §7 实验清单。

### 4.3 进阶:可学习权重(Kendall et al. 2018)

```
L = Σ_i (1/2ŝ_i²)·L_i + log ŝ_i
```

多任务不确定性自动加权。**注意**与 heteroscedastic head 的 aleatoric 方差
概念重叠,需仔细避免冲突 —— 列为 v2.1 进阶,首版用 σ 归一化 + 手调 λ。

---

## 5. 空间平滑(唯一保留的正则)

### 5.1 为什么不做时间平滑

monthly anomaly 的月际变化**可以且应当大**(尤其 fCO2/DIC):
ENSO 遥相关、生物泵年际波动、垂直混合事件都是真实月际信号;对它们强加
时间平滑会**抹掉真实信号**,引入系统性偏差。因此:

- **空间平滑保留**(海洋化学场空间上由扩散主导,平滑有物理依据);
- **时间平滑废弃**(默认不加;进阶可选 AR(1) 状态先验,见 §5.3)。

### 5.2 空间平滑实现:二阶 Laplacian + 沿岸加权

```python
def spatial_smooth(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """二阶差分(离散 Laplacian);weight 可沿岸加权。x: [..., H, W]"""
    lap_x = x[..., 1:-1, 2:] - 2 * x[..., 1:-1, 1:-1] + x[..., 1:-1, :-2]
    lap_y = x[..., 2:, 1:-1] - 2 * x[..., 1:-1, 1:-1] + x[..., :-2, 1:-1]
    sq = lap_x.square() + lap_y.square()
    if weight is not None:
        sq = sq * weight[1:-1, 1:-1]
    return sq.mean()
```

- 权重 `w(x) = 1/(1 + dist_to_land)`:近岸(河口、上升流)允许真实强梯度,
  开阔洋要求平滑;
- 作用对象:模型输出的网格场 **TÂ, DIĈ, SSŜ**(fCO2 可选);
- 梯度经线性差分自然反传,可微;
- λ_smooth 小(0.01–0.1),防噪声过拟合而非主导回归。

### 5.3 进阶:AR(1) 时间先验(默认关闭)

把"时间连续性"从**硬正则**改为**可学习先验**,让数据决定连续性:

```
anomaly_t = ρ · anomaly_{t-1} + ε_t,    ε ~ N(0, σ²)
ρ 每输出变量一个可学习参数(SST 学出 ρ≈0.9,fCO2 学出 ρ≈0.2)
```

- 平滑度由数据决定,而非拍脑袋;
- 与 heteroscedastic head 的方差天然衔接(σ 承载"该月异常多大");
- 首版默认 `off` —— transformer 的时间 attention + 输入场连续性
  已隐式处理时间维,显式正则反而可能有害。

---

## 6. 损失最终形态

```
L = λ_fCO2 · mean( (fCO2̂ − fCO2_SOCAT)² / σ_fCO2² )
  + λ_SSS  · mean( (SSŜ − sal_SOCAT)²  / σ_SSS² )
  + λ_TA   · mean( (TÂ  − TA_GLODAP)²  / σ_TA² )
  + λ_DIC  · mean( (DIĈ − DIC_GLODAP)² / σ_DIC² )
  + λ_chem · mean( (CO2SYS(T,S,TÂ,DIĈ) − fCO2_SOCAT)² / σ_chem² )
  + λ_smooth · spatial_smooth(TÂ, DIĈ, SSŜ)

默认 λ: {fCO2:1, SSS:1, TA:5, DIC:5, chem:1, smooth:0.05}
默认 σ: {fCO2:5, SSS:0.1, TA:8, DIC:8, chem:5}
```

- 全部无量纲、可解释;
- 6 个 λ + 5 个 σ 均入 config,可扫描;
- 主任务 fCO2 同时受 L_fCO2(直接)与 L_chem(化学闭合)双约束。

---

## 7. 实现与验证清单

1. **GLODAP 锚点提取**:v2.2023 表层 TA/DIC(深度 <10 m),转 0.125° 格点,
   与 SOCAT 时间窗(1993–2026)对齐,输出锚点表(含 σ_TA/σ_DIC);
2. **σ 归一化**:实现 `noise_normalized_loss(residual, sigma)` 工具函数;
3. **batch 过采样**:GLODAP 锚点与 SOCAT 点混合的样例采样器;
4. **λ/σ config**:`configs/*.yaml` 增 `joint_loss:` 节;
5. **灵敏度扫描**:λ 网格 {0.1,1,10},监控测试 R²/RMSE + 化学自洽残差;
6. **消融**:去掉 L_chem / L_TA / L_SSS 各跑一次,量化每项贡献;
7. **验证指标**:三段(train/val/test)R²/RMSE、TA/DIC 与 GLODAP 锚点吻合度、
   CO2SYS 自洽残差分布、产品异常月际变率是否保留(防过度平滑)。

---

## 8. 与本仓库现有组件的对接

| 组件 | 角色 |
|------|------|
| `recad.data.binning.GriddedTarget` | fCO2/SSS/SST 网格场 + std_err(σ 来源) |
| `recad.data.tracks.TrackData` | SOCAT 散点 + sal/temp 现场字段 |
| `recad.data.split.split_points` | 航次级 train/val + 2004–2005 整年 test |
| `recad.utils.chem`(PyCO2SYS) | CO2SYS 前向(fCO2 由 TA/DIC 计算)与化学闭合 |
| `recad.model.st_transformer` | 骨干输出:多任务头(fCO2̂/SSŜ/TÂ/DIĈ) |
| `recad.train.losses` | 新增 joint loss 组合器(本文档实现) |
| `third_party/ESPER`(Carter) | TA_SST-SSS 先验(δTA 残差设计的锚,可选) |

> ESPER(Brendan Carter, NOAA)含 TA 经验估算(线性/神经网络),可作为
> TA 先验 `TA_Carter(SSS,SST,lon,lat)`,让模型学 **δTA = TÂ − TA_Carter**(残差),
> GLODAP 只校准残差 —— 数据需求进一步骤降。详见 `third_party/ESPER/README.md`。
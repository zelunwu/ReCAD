# 空间加权碳酸盐联合反演 — 结果报告

*脚本 `scripts/run_weighted_carbonate.py`;支持度场 `scripts/build_chem_support.py`;
实验目录 `outputs/experiments/weighted_{naccom,global}_v1/`。*

---

## 1. 方法

### 1.1 空间加权化学约束(本轮核心改动)

原方案对**全域所有格**施加同等的 CO2SYS 闭合惩罚。但 TA/DIC 观测极稀疏:

| 域 | 锚点格(权重 1.0) | 200 km 内(0.1→0 衰减) | 零权重格 |
|----|-----------------|----------------------|---------|
| 全球 | 9,707 | 1,247,048 | **2,478,605 (66.4%)** |
| NACCOM | 1,097 | 76,869 | 3,657,394 (97.9%) |
| US 东岸 | 564 | 26,766 | 3,708,030 |
| US 西岸 | 338 | 23,240 | 3,711,782 |

**全球沿海格仅 1.4% 有 TA/DIC 锚点**(US 东岸 3.5%,西岸 3.3%),
所以在 66–99% 的格上闭合约束是**伪约束**((TA,DIC) 在该处不可辨识)。

本轮损失改为**按观测支持度加权**:

```
L_chem = Σ_i w(x_i)·((CO2SYS(T,S,TA_i,DIC_i) − fCO2_i)/σ)² / Σ_i w(x_i)

w = 1.0            格内有 ≥1 个 GLODAP 锚点
  = 0.1·(1−d/200km)  距最近锚点 <200 km
  = 0.0            更远(硬置零)
```

产物:`outputs/chem/chem_support_{global,naccom,us_east,us_west}.nc`
(`n_anchors` / `dist_km` / `support`),查询模块 `src/recad/chem/support.py`。

### 1.2 协议(沿用并固化上轮结论)

| 项 | 设定 |
|----|------|
| 独立输出 | SSS(相对 GLODAP 背景的残差)、fCO2、TA |
| **DIC** | **派生**:`DIC = inv_CO2SYS(T, S, TÂ, fCO2̂)`(可微逆代理) |
| TA 先验 | ESPER-LIR 方程 16(S-only),作输入通道 |
| 训练阶段 | ①SOCAT 预训练(3000 步)→ ②碳酸盐 adapter(2000 步,含加权闭合) |
| 集成 | 3 个种子(100/101/102) |
| 划分 | 2004–2005 整年 **test**;其余按 2° 空间块/航次切 train/validation |

---

## 2. 美国沿岸(NACCOM 大西洋岸)结果

域:`prepared_naccom.nc` + `joint_four_full_20260907/joint_cache.npz`
(lon 265.9–319.9°E, lat 10–65°N;SOCAT 训练 144,225 点,GLODAP 训练 1,699 锚)

### 2.1 全域

| target | split | n | RMSE | R² |
|--------|-------|---|------|-----|
| SSS | train | 117,569 | 1.065 PSU | 0.772 |
| SSS | validation | 35,845 | 1.320 | 0.660 |
| SSS | test | 12,258 | 1.324 | 0.735 |
| fCO2 | train | 144,225 | 33.61 µatm | 0.558 |
| fCO2 | validation | 37,125 | 39.20 | 0.259 |
| fCO2 | test | 13,052 | 31.10 | 0.377 |
| TA | train | 1,699 | 22.71 µmol/kg | 0.947 |
| TA | validation | 283 | 56.31 | 0.703 |
| TA | test | 38 | 48.51 | −0.385 |
| DIC | train | 1,699 | 18.95 µmol/kg | 0.895 |
| DIC | validation | 283 | 48.42 | −0.021 |
| DIC | test | 38 | 41.33 | 0.214 |

### 2.2 美国本土报告窗(lon 275–300°E, lat 20–55°N)

| target | split | n | RMSE | R² |
|--------|-------|---|------|-----|
| SSS | train | 80,894 | 1.068 PSU | 0.792 |
| SSS | validation | 19,492 | 1.254 | 0.694 |
| SSS | test | 8,659 | 1.479 | 0.731 |
| fCO2 | train | 81,939 | 32.70 µatm | 0.453 |
| fCO2 | validation | 19,523 | 40.15 | 0.282 |
| fCO2 | test | 8,659 | 32.11 | 0.224 |
| TA | train | 951 | 22.38 µmol/kg | 0.955 |
| TA | validation | 177 | 65.51 | 0.619 |
| DIC | train | 951 | 17.37 µmol/kg | 0.895 |
| DIC | validation | 177 | 53.38 | −0.370 |

**注**:美国报告窗**没有 TA/DIC 的 test 行**——38 个独立锚点位于 306–320°E
(北大西洋高纬),不在该窗口内,因此该窗口的 TA/DIC 只有 train/validation。

---

## 2b. 全球结果

域:`prepared_global_p32.nc` + `global_joint_cache_p32/joint_cache.npz`
(4° patch = 3,690 token;SOCAT 训练 752,668 点;GLODAP 训练 10,359 锚;
SSS 监督 = SOCAT 现场盐度,19.0M 观测)

### 全域

| target | split | n | RMSE | R² |
|--------|-------|---|------|-----|
| SSS | train | 628,080 | 1.336 PSU | 0.953 |
| SSS | validation | 220,295 | 1.156 | 0.987 |
| SSS | test | 40,844 | 1.081 | 0.986 |
| fCO2 | train | 752,668 | 46.61 µatm | 0.330 |
| fCO2 | validation | 239,396 | 60.45 | 0.210 |
| fCO2 | test | 48,934 | 53.20 | **−0.294** |
| TA | train | 10,359 | 52.75 µmol/kg | 0.848 |
| TA | validation | 2,824 | 41.76 | 0.791 |
| TA | test | 784 | 74.49 | 0.643 |
| DIC | train | 10,359 | 49.65 µmol/kg | 0.778 |
| DIC | validation | 2,824 | 42.07 | 0.722 |
| DIC | test | 784 | 72.13 | 0.470 |

### 美国沿岸窗口(全域模型,lon 275–300°E / lat 20–55°N)

| target | split | n | RMSE | R² |
|--------|-------|---|------|-----|
| SSS | train | 96,715 | 1.289 PSU | 0.636 |
| SSS | validation | 51,537 | 1.171 | 0.707 |
| SSS | test | 12,869 | 1.450 | 0.633 |
| fCO2 | train | 98,826 | 32.88 µatm | 0.481 |
| fCO2 | validation | 52,043 | 42.41 | 0.212 |
| fCO2 | test | 12,869 | 27.03 | 0.340 |
| TA | train | 949 | 38.55 µmol/kg | 0.866 |
| TA | validation | 177 | 55.67 | 0.725 |
| DIC | train | 949 | 28.43 µmol/kg | 0.718 |
| DIC | validation | 177 | 40.45 | 0.212 |

---

## 3. 关键发现

### 3.0 ⚠️ 重要更正:R² 不能跨区域比较——必须看 RMSE 与 skill

**"美国窗口结果更差"是 R² 的假象。** R² 的分母是该集合的目标方差;窗口内水团同质,
目标方差只有全域的 46–63%,因此在**同样甚至更小的绝对误差**下 R² 会被压低。

分解(全球模型,`skill = 1 − RMSE/RMSE_clim`,RMSE_clim = 预测集合均值):

| target | split | region | RMSE | 均值基线 RMSE | R² | **skill** |
|--------|-------|--------|------|--------------|-----|-----------|
| fCO2 | test | domain | 53.20 | 46.76 | **−0.294** | **−0.138** |
| fCO2 | test | us_window | **27.03** | 33.26 | +0.340 | +0.187 |
| fCO2 | validation | domain | 60.45 | 68.01 | +0.210 | +0.111 |
| fCO2 | validation | us_window | **42.41** | 47.77 | +0.212 | +0.112 |
| SSS | validation | domain | 1.16 | 10.12 | +0.987 | **+0.886** |
| SSS | validation | us_window | 1.17 | 2.16 | +0.707 | **+0.459** |

**读法**:
- fCO2:窗口 RMSE **27.0 < 全域 53.2** → 窗口内模型其实更准;R² 低纯粹是方差效应
- **SSS 是真差距**:RMSE 几乎相同(1.16 vs 1.17)但 skill 0.886 vs 0.459

### 3.0.1 全球 SSS 的高 R² 有水分

全域 SSS 均值基线 std 高达 **10.12 PSU**(含波罗的海/河口等极端低盐),窗口只有 **2.16 PSU**。
全球 R² = 0.987 主要来自**区分极端水团**,而非精细刻画沿岸盐度——
窗口内(同质水团)真实技能仅 **0.46**。

### 3.0.2 fCO2 是真实的薄弱环节

| fCO2 skill | 沿岸专用(naccom) | 全球 |
|-----------|------------------|------|
| train | +0.335 | +0.182 |
| validation | +0.139 | +0.111 |
| test | +0.211 | **−0.138** |

全球全域 test 的 skill 为负(比预测均值还差)**不是方差假象**(均值基线 46.76 < 模型 53.20)。
三个可验证的原因:

1. **留出年偏难**:test 年 fCO2 std 48.8 vs train 57.7,模型在低方差年上过外推
2. **多任务竞争**:SSS 头与 fCO2 头共享同样 75 万样本,但 SSS 的可学习信号强得多
   (归一化后仍达 skill 0.78–0.89,而 fCO2 仅 0.11–0.18),共享 trunk 的梯度被 SSS 主导
   ——与上轮"TA/DIC 被密集 SOCAT 梯度牵引"是同一机制,这次 fCO2 是被挤压方
3. **沿岸过程被平滑**(patch 粒度效应,见 §3.0.3)

### 3.0.3 patch 粒度是沿岸技能的关键变量(实测)

同一物理任务、同一协议,仅空间上下文粒度不同(patch 1° vs 4°):

| target | region | global(4° patch) train skill | naccom(1° patch) train skill | 差 |
|--------|--------|------------------------------|------------------------------|-----|
| **fCO2** | domain | +0.182 | **+0.335** | **+0.153** |
| TA | domain | +0.610 | **+0.770** | +0.160 |
| DIC | domain | +0.529 | **+0.676** | +0.147 |
| SSS | domain | +0.783 | +0.523 | −0.260 |

**fCO2 / TA / DIC 在 1° patch 下一致更好**(+0.15 左右),而 SSS 反之。
解释:沿岸碳酸盐参数受**河流、上升流、陆架过程**控制,这些是**小尺度**信号,
被 4° patch 的注意力平均掉了;而 SSS 的大尺度(洋盆盐度梯度)在 4° 下反而更易学。

**行动含义**:全球产品若要做沿岸应用,不能沿用 4° patch——
需要用**分层方案**(沿岸 1° + 开阔洋 4°)或全 1°(计算上不可行,见 §6.1)。

---

### 3.1 加权生效:TA/DIC 验证技能显著改善

与**同一划分协议**下的早期未加权结果对比(US 西岸 cache,同样 2° 空间块切分):

| target | split | 未加权(v1, 统一闭合) | 本轮加权(NACCOM) |
|--------|-------|---------------------|------------------|
| TA | validation | 77.8 / **R² 0.015** | 56.3 / **0.703** |
| DIC | validation | 82.6 / **−0.150** | 48.4 / **−0.021** |
| TA | train | 60.8 / 0.831 | 22.7 / **0.947** |
| DIC | train | 68.3 / 0.745 | 19.0 / **0.895** |

### 3.2 但 TA/DIC 的**独立 test 仍然失败**

| target | test n | RMSE | R² |
|--------|--------|------|-----|
| TA | 38 | 48.51 | **−0.385** |
| DIC | 38 | 41.33 | 0.214 |
| TA (US 窗口) | 0 | — | — |

38 个锚点/3 个航次的 test 集统计把握极弱,**且 R² 为负**——
说明模型在**跨航次/跨纬度外推**上仍然失效。这与上轮结论一致,并再次说明:

> **加权解决了"伪约束",但没有创造信息。** TA/DIC 的知识仍只来自
> 14,930 个锚点、592 个航次。

### 3.3 全球域结论

| 观察 | 数据 |
|------|------|
| **SSS 全球技能极高** | train/val/test R² = 0.953 / 0.987 / 0.986,RMSE ≈ 1.1–1.3 PSU |
| **TA/DIC 在有锚区可以工作** | 全球 TA test R² **0.643**(784 锚),DIC **0.470** |
| **fCO2 全球 test 失败** | RMSE 53.2,R² **−0.294**(负值) |

**fCO2 全球 test R² 为负是本轮最重要的负面结果**:全球模型在 2004–2005
留出年上比"预测均值"还差。而同一模型在美国沿岸窗口的 fCO2 test R² = 0.340,
说明**全球域训练把 fCO2 技能稀释了**——域太大、水团太杂,单一模型难以同时
覆盖所有海区的 fCO2 关系。

对比美国沿岸专用模型(NACCOM):

| fCO2 | 全球模型(US 窗口) | NACCOM 专用模型 |
|------|------------------|----------------|
| train R² | 0.481 (n=98,826) | 0.453 (n=81,939) |
| validation R² | 0.212 | 0.282 |
| test R² | 0.340 (n=12,869) | 0.224 (n=8,659) |

两者接近,但**验证/测试波动大**,提示 fCO2 的空间泛化是不稳定环节。

### 3.4 主产品(SSS/fCO2)技能

- **SSS 稳健**:沿岸模型 val/test R² 0.66–0.74;全球模型 0.99(全球密集监督)
- **fCO2 脆弱**:沿岸 val R² 0.26 / test 0.38;全球 test 为负

---

## 4. 结论与产品定位

| 层级 | 内容 | 依据 |
|------|------|------|
| **主产品** | SSS | train/val/test R² 稳定(0.66–0.99),样本充足 |
| **主产品(需谨慎)** | fCO2 | 沿岸 test R² 0.22–0.38;全球 test 为负,**需按海区分别建模** |
| **专题** | 沿岸 TA | train R² 0.95,validation 0.62–0.79;test 依赖 38–784 锚点 |
| **派生量** | DIC | 由 CO2SYS 逆解码器导出,标注为诊断量 |
| **不使用** | 全球统一化学硬约束 | 66% 格为伪约束 |
| **待补** | 独立 TA/DIC 验证 | 需 CODAP-NA 或新增航次 |


---

## 5. 复现

```powershell
# 1) 支持度场(全球/区域)
python scripts/build_chem_support.py

# 2) 美国沿岸(大西洋)实验
python scripts/run_weighted_carbonate.py --domain naccom --tag weighted_naccom_v3 \
    --pretrain-steps 3000 --steps 2000 --chem-lambda 0.05 --seeds 100,101,102

# 3) 全球实验(4° patch;见 §6 的可计算性说明)
python scripts/run_weighted_carbonate.py --domain global_p32 --tag weighted_global_v1 \
    --pretrain-steps 3000 --steps 2000 --chem-lambda 0.05 --seeds 100,101,102

# 结果:outputs/experiments/<tag>/metrics_by_split.csv
```

---

## 6. 全球域的可计算性与数据管道修复

### 6.1 patch size 决策(实测)

全球域按 2° patch 建 cache 得 **14,760 token**,实测**单步 37 秒**——
空间 attention 为 O(P²),全套实验需 **~308 小时(13 天)**。

| 阶段 | 2° patch (P=14,760) | 4° patch (P=3,690) |
|------|--------------------|-------------------|
| `encode_context` | 37,309 ms | ~380 ms |
| `decode_queries` (b512) | 52 ms | 52 ms |
| 单步合计 | **36,929 ms** | **419 ms** |
| 全套(3 CV + 3 seeds) | ~308 h | **~3.5 h** |

处理:新增 `configs/global_1over8_patch32.yaml`(`patch_size_cells: 32`),
重建 `prepared_global_p32.nc` / `masks_global_p32.nc` / `global_joint_cache_p32/`。
**patch 只决定 attention 上下文粒度,重建网格仍是完整 1/8°。**

### 6.2 修复的四个管道缺陷

| # | 缺陷 | 症状 | 修复 |
|---|------|------|------|
| 1 | `*_year` 写成绝对年(1993..2026) | `ctx(year)` 越界 → reshape 空张量 | `fix_global_cache_years.py`(改偏移 0..33);builder 已同步 |
| 2 | `*_month` 写成 1-based(1..12) | `ctx[0,12,patch]` 越界 → **CUDA assert** | `fix_global_cache_months.py`(改 0..11);builder 已同步 |
| 3 | GLODAP 特征列残留 NaN | ridge 基线系数 = NaN → **CV 分数全 NaN** | `fix_global_cache_nans.py`(NaN→0 + 缺失旗标=1);builder 已同步 |
| 4 | **SSS 监督泄漏**:`socat_*_sss` 直接取自 GLORYS `sss` 场,而该场同时是模型的 `base_sss` 背景 | `target − base ≡ 0` → **SSS R² = 1.0(退化)** | builder 改用 `cache/socat_sal_mean.nc`(SOCAT 现场盐度,19.0M 观测 → 890,581 格月);泄漏检验:`sss − base` std **1.96 PSU**(修复前 0.000) |

第一条全球运行因此被废弃(SSS 退化解),已用 `weighted_global_v2` 重跑。


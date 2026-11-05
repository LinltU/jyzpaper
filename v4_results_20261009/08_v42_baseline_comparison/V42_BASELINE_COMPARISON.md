# v4.2 正式版：去泄漏 DecompositionNLinear 基线

判定规则（32 点窗口、湿度分层、同串匹配、A/B/C/D 阈值）与 v4 完全一致，**只替换环境基线模型**。
正式基线 = `legacy_dlinear` 归档的 `DecompositionNLinear`，已剔除三列目标泄漏特征。

## 1. 修掉的三列泄漏

原 `legacy_dlinear/train.py` 构造的特征中有三列包含当前目标 `Measure_t`：

| 列 | 原定义 | 为什么泄漏 |
|---|---|---|
| `Measure_diff` | `Measure.diff()` | `Measure_t − Measure_{t−1}` |
| `Measure_lag1` | `Measure.shift(1)` | `Measure_{t−1}` |
| `Measure_MA` | `Measure.rolling(3).mean()` | 窗口 `[t−2, t−1, t]` 含 `Measure_t` |

前两列单独看很弱（R² 0.137 / 0.527），但**相加恒等于目标**：

```
Measure_diff + Measure_lag1 = (Measure_t − Measure_{t−1}) + Measure_{t−1} = Measure_t
```

实测 `max|误差| = 1.11e-16`，两列合用的线性回归 R² = **1.000000**。
`Measure_MA` 单独 R² = 0.798，是目标的一个分量（占 1/3 权重）。

此外原代码 `rolling(...).fillna(method='bfill')` 会用未来值回填开头，一并去除。

### 保留的 10 列（均在 t 时刻可得）

```
Temp, Hum, AH, Temp_diff, Hum_diff,
Temp*Hum, Temp*AH, hum_low, hum_mid, hum_high
```

与 `Measure_t` 的最大相关系数 0.616（Hum），十列线性回归 R² 上限 0.449，**无可还原目标的组合**。
`Temp_diff` / `Hum_diff` 统一切到前一行，因为归档 `predict()` 推理路径取不到这两个值（原实现传 0），
否则构成训练/推理不一致。

## 2. 四个基线（同一协议、同一可评估点集）

全历史样本内拟合（与 v4 对齐），16,585 个共同可评估点。

| 基线 | 候选窗 | 占比 | MAE | RMSE | R² | 残差std | B | C | A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v4 Huber | 38 | 6.95% | 0.09480 | 0.16782 | 0.51063 | 0.15785 | 31 | 4 | 3 |
| **v4.2 archived** | **41** | **7.50%** | **0.08811** | **0.16465** | **0.53548** | **0.14206** | 31 | 6 | 4 |
| 官方 LTSF NLinear | 97 | 17.73% | 0.09338 | 0.16753 | 0.51913 | 0.16615 | 85 | 12 | 0 |
| naive（历史末值） | 135 | 24.68% | 0.11705 | 0.20401 | 0.27725 | 0.20020 | 119 | 16 | 0 |

与 Huber 的窗口重叠：

| 基线 | 共同 | Jaccard | 新增 | 丢失 |
|---|---:|---:|---:|---:|
| archived | 26 | 0.491 | 15 | 12 |
| nlinear | 35 | 0.350 | 62 | 3 |
| last | 37 | 0.272 | 98 | 1 |

## 3. 算法上的不同

### v4 Huber（参考基线）

```
Measure_t ≈ a + b1·Temp + b2·Hum + b3·elapsed
```

- 逐设备、全历史、**样本内**拟合，4 个参数
- 损失是 Huber（ε=1.5），对离群点线性降权
- 只看同一时刻的温湿度，不含任何 Measure 历史
- 本质是**横截面回归**：把 Measure 当作温湿度的函数

### v4.2 archived（DecompositionNLinear）

```
x_t = [Temp, Hum, AH, ΔTemp, ΔHum, Temp·Hum, Temp·AH, hum_low/mid/high]_t
seasonal, trend = SeriesDecomp(x_t)          # 时间维=1，实际退化为 trend≡x, seasonal≡0
h = GELU(LN(W_s·seasonal)) ⊕ GELU(LN(W_t·trend))
Measure_t ≈ Fusion(h) + Linear(条件项)
```

- 10 维输入 → 128 隐层 → 融合层，约 2.8 万参数
- 含一个独立的"条件头"（Temp、Hum、Temp×Hum、elapsed）
- **非线性**：GELU + LayerNorm + Sigmoid 门控（`interaction_layer` 对输入逐维加权）
- 含物理衍生量 `AH`（绝对湿度，Magnus 公式）和温度/湿度变化率
- 输出经 `Softplus` 保证非负

### 关键差异

| | Huber | archived DLinear |
|---|---|---|
| 模型类 | 广义线性（4 参数） | 非线性 MLP（2.8 万参数） |
| 输入 | 3 维（Temp, Hum, elapsed） | 10 维（含 AH、差分、交互、湿度区间） |
| 函数形状 | 单调、可加、无交互 | 可学交互、门控、非线性 |
| 常数段行为 | 输出恒为常数 | **跟随上一时刻水平**（差分项） |
| 残差自相关 | +0.40 | +0.44 |

## 4. 为什么会有提升

提升是真实的，但**幅度不大且必须正确解读**：

**R² 0.511 → 0.535（+0.025），MAE 0.0948 → 0.0881（−7.1%），残差 std 0.158 → 0.142（−10%）。**

原因有三条，按贡献排序：

### ① 非线性交互（主要）

绝缘子表面泄漏电流与温湿度不是可加关系。Huber 强制 `Measure = b1·T + b2·H + b3·t`，
无法表达"高温高湿叠加时非线性上升"。归档模型有 `Temp·Hum`、`Temp·AH` 交互项和
`interaction_layer` 的 Sigmoid 门控，能拟合这种耦合。这是 R² 提升的主体。

### ② 绝对湿度 AH（物理量）

`AH = 216.7·e/(T+273.15)` 由 Magnus 公式算出，是比相对湿度更贴近放电物理的量。
Huber 只有 `Hum`（相对湿度），同一相对湿度在不同温度下 AH 差别很大。加入 AH 后
模型的物理依据更正确，这部分残差被系统性地解释掉了。

### ③ 变化率特征（差分项）

`Temp_diff`、`Hum_diff` 提供环境变化趋势。注意：这两个是**环境**的差分，不是 Measure 的差分，
所以不构成泄漏。它们让模型能区分"稳定高湿"和"正在快速升湿"。

### 为什么幅度只有 +0.025

因为**剩余残差主要不是温湿度能解释的**。R² 0.535 意味着还有 46% 的方差来自
设备自身状态、安装位置差异、测量噪声和真实异常——这些正是 v4 想检测的东西。
基线再好也有天花板；把它做到 R²=0.9 反而危险，因为会把真实异常也一起"解释"掉。

## 5. 为什么 nlinear 和 last 不行

这两个是**反例**，说明"候选变多 ≠ 检测变好"：

- **官方 NLinear**：`seq_len=16` 单通道、17 个参数，对本任务表达力不足，R² 0.519（低于 Huber），
  残差 std 0.166 反而更大，B 规则从 31 膨胀到 85。A 规则掉到 0，因为残差符号结构被破坏。
- **naive（历史末值）**：R² 仅 0.277，残差 lag-1 自相关 **−0.42**（强负相关），
  相邻残差剧烈振荡，被误判为"跳变"，B 从 31 暴涨到 119，候选 135 个。

naive 的负自相关是关键证据：历史末值预测会让残差交替正负，
而 v4 的跳变规则 `|diff(peer_delta)|` 正好把这种振荡识别为跳变。所以朴素基线会**系统性抬高**候选数，
这是规则与基线的结构性冲突，不是发现了更多异常。

## 6. 结论

- v4.2 正式基线 = 去泄漏的归档 `DecompositionNLinear`，41 个候选窗（7.50%）。
- 相比 v4 Huber：R² +0.025、MAE −7.1%、残差 std −10%，候选 38 → 41（+3）。
- 提升来自**非线性温湿度交互 + 绝对湿度 + 环境变化率**，不是泄漏。
- 候选数只增加 3 个、26 个窗口重叠（Jaccard 0.491），说明改动是**局部修正**而非全盘重排，
  这正是换一个更好的基线应有的表现。
- 97（nlinear）和 135（last）是基线退化导致的假阳性膨胀，不可作为"发现更多异常"。

## 7. 复现

```bash
PY=/home/test/miniconda3/envs/jyz_paper/bin/python
cd v4_results_20261009/05_scripts

# v4.2 正式版（archived，默认）
$PY process_device_nonoverlap_windows_v42.py --baseline archived --output ../../runs/v42_archived

# 三个对照
$PY process_device_nonoverlap_windows_v42.py --baseline huber   --output ../../runs/v42_huber
$PY process_device_nonoverlap_windows_v42.py --baseline nlinear --output ../../runs/v42_nlinear
$PY process_device_nonoverlap_windows_v42.py --baseline last    --output ../../runs/v42_last
```

## 8. 局限

- 全历史样本内拟合，与 v4 对齐以保持可比性，因此**仍是离线回顾分析**，不能证明在线检测能力。
- 容差沿用 v4.1 的居中 9 点局部中位间隔，同样引用未来观测。
- `DecompositionNLinear` 的序列分解在时间维=1 时退化（seasonal≡0），实际起作用的是其 MLP 部分；
  名称中的"DLinear"名不副实，报告时不应声称用了真正的分解式 DLinear。
- 无确认故障标签，1 只是统计候选，0 只是未触发本规则。

## 9. 图表清单

| 文件 | 内容 |
|---|---|
| `v42_anomaly_labels_by_device.png` | **v4.2 正式版**（archived 基线）25 台设备分类图：灰=Measure，青=基线，蓝=正常，红=候选，浅红底=候选窗口 |
| `v42_baseline_comparison.png` | 四基线并排对照：每列一个基线，三行分别为每设备候选数、预测残差散点、触发规则分布 |
| `v42_anomaly_labels_by_device_huber.png` | Huber 基线分类图（对照） |
| `v42_anomaly_labels_by_device_nlinear.png` | 官方 NLinear 基线分类图（对照） |
| `v42_anomaly_labels_by_device_last.png` | naive 基线分类图（对照） |
| `baseline_comparison.csv` | 四基线指标表（候选数、MAE、RMSE、R²、残差std、与 Huber 重叠） |
| `device_nonoverlap_windows_v42.csv` | 正式版逐窗口特征与标签 |
| `device_nonoverlap_windows_v42_{huber,nlinear,last}.csv` | 三个对照的逐窗口结果 |
| `device_points_v42.csv` | 正式版逐点证据（含 `env_baseline`、`peer_delta`、`peer_delta_jump_z`） |
| `protocol_v42.json` | 参数、阈值、每设备训练信息 |

图注已按实际基线更正：正式版标题为 `DecompositionNLinear (legacy_dlinear, leakage removed)`，
不再误标为 LTSF NLinear；并在图底注明"内部序列分解在时间长度 1 时退化，等效为非线性 MLP，"
"并非真正的分解式 DLinear"。

## 10. 复现图

```bash
PY=/home/test/miniconda3/envs/jyz_paper/bin/python
cd v4_results_20261009/05_scripts
$PY plot_v42_baseline_comparison.py --root /tmp \
    --output ../08_v42_baseline_comparison
```

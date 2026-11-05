# v4-evi：当前推荐方案（SOTA 数据增强）

本目录 `03_benchmark_v4_evidence_features/` 是当前三套候选中**唯一在每个比例、两种分类器上都获得正增强收益**的配置。

## 1. 为什么选 v4-evi

三套配置共用同一批生成器、下游分类器、比例和种子：

| 配置 | 标签 | 候选窗 | 特征集 |
|---|---|---:|---|
| base | v4（Huber） | 38 | Measure, env_residual, Hum, Temp, delta_measure |
| **v4-evi** | **v4（Huber）** | **38** | **Measure, env_residual_z, peer_delta_z, peer_delta_jump_z, delta_measure** |
| v4.2-evi | v4.2（archived DecompositionNLinear） | 40 | v4-evi |

### Real-only AP（无增强的基线）

| 比例 | 分类器 | base | **v4-evi** | v4.2-evi |
|---|---|---:|---:|---:|
| 2:8 | HGB | 0.1217 | 0.2062 | 0.2294 |
| 2:8 | RF | 0.1432 | 0.1625 | 0.1720 |
| 3:7 | HGB | 0.1716 | 0.2071 | 0.2494 |
| 3:7 | RF | 0.1948 | 0.1884 | 0.2057 |
| 4:6 | HGB | 0.1764 | **0.2746** | 0.2285 |
| 4:6 | RF | 0.1831 | **0.2428** | 0.1875 |
| 5:5 | HGB | 0.2149 | 0.2323 | 0.2425 |
| 5:5 | RF | 0.1893 | **0.2294** | 0.1769 |

### 增强优于 Real-only 的比例数

| 配置 | HGB | RF |
|---|---|---|
| base | 3/4 | 4/4 |
| **v4-evi** | **4/4** | **4/4** |
| v4.2-evi | 3/4 | 4/4 |

v4-evi 是唯一 8/8 全为正增益的配置，且在 4:6 和 5:5 上的 AP 绝对值最高。

## 2. 核心方法

### 特征集

```
Measure            当前泄漏电流测量值
env_residual_z     环境残差（温湿度/时间 Huber 基线）的稳健 z
peer_delta_z       同串 peer 偏离（环境残差之差）的分层 z
peer_delta_jump_z  同串 peer 偏离的一阶跳变 z（裁剪到 20）
delta_measure      相邻 Measure 差分
```

关键在 `peer_delta_z` 与 `peer_delta_jump_z`：v4 的 38 个候选中 31 个由 peer 跳变触发，
而 base 特征完全看不到 peer 信息，所以分类器无从识别。加入这两个通道后，
2:8 的 Real-only AP 从 0.1217 提升到 0.2062（**+0.085**），是三套对比中最大的一步。

**特征的影响远大于标签基线的影响（约 3~4 倍）。**

### 标签

沿用 v4 时间/湿度条件化规则：32 点无重叠窗口、逐设备 Huber 环境基线、
湿度分箱秩正态标准化、同串（intra-string）peer 中位数比较、A/B/C/D 四规则。

- 候选窗口 38 / 547（6.95%）
- 点级异常率 7.10%，窗口级 7.68%

### 实验协议

| 项目 | 取值 |
|---|---|
| 窗口长度 | 4 点滑动；点落在候选窗内即异常，窗口含任一异常点为正类 |
| 训练:测试 | 2:8, 3:7, 4:6, 5:5（逐设备时间切分） |
| 生成方法 | 18 个（CSDI, MTSCI, IMDiffusion, Diffusion-TS, D3R + 13 个机制移植） |
| 下游分类器 | RF / HGB，阈值在 validation 按 F1 选取 |
| 训练配置 | 60 epoch / 20 diffusion steps / 种子 42, 123, 2024 |
| 分组 scope | peer_group（intra-string，9 组：7 组 3 台 + 2 组 2 台） |

## 3. 逐比例最佳方法（HGB）

| 比例 | 最佳方法 | AP | Δ vs Real-only |
|---|---|---:|---:|
| 2:8 | CoSTI | 0.2298 | +0.0237 |
| 3:7 | MTSCI | 0.2208 | +0.0137 |
| 4:6 | IMDiffusion | 0.2786 | +0.0040 |
| 5:5 | MTSCI | 0.2639 | +0.0316 |

RF 侧：CoSTI（2:8, +0.0625）、CoSTI（3:7, +0.0122）、NsDiff（4:6, +0.0260）、SSD-TS（5:5, +0.0469）。

**注意：方法排名不稳定。** 同为 5:5 HGB，base 的最佳是 Diffusion-TS，v4-evi 是 MTSCI，
v4.2 是 CSDI。方法间差异小于种子波动，不应据此对生成器排名。

## 4. 文件

| 文件 | 内容 |
|---|---|
| `summary.csv` | 152 行：比例 × 方法 × 分类器的均值/标准差 |
| `all_results.csv` | 逐次原始结果（比例 × 方法 × 种子 × 分类器） |
| `formal_v4_summary_mean_std.csv` | 正式汇总（均值 ± 标准差） |
| `formal_v4_best_methods.csv` | 每比例最佳方法与 Real-only 对比 |
| `formal_v4_annotation_by_device.csv` | 逐设备候选标注统计 |
| `formal_v4_annotation_by_group.csv` | 逐串组候选标注统计 |
| `formal_v4_raw_quality_audit.csv` | 生成数据质量审计 |
| `protocol.json` | 完整参数、标签统计 |
| `v4_window_labels.csv` | 38 个候选窗标签 |
| `training_audit.csv` | 每个 scope 的训练样本数、生成数、末轮损失 |
| `DEVICE_DATA_V4_PAPER_AUG_RESULTS.md` | 本配置原始报告 |
| `COMPARE_V4FEATURE.md` | base vs v4-evi 特征对比 |
| `figures/method_ap_heatmap.png` | 方法 × 比例 AP 热力图 |
| `figures/real_only_vs_best_paper_augmentation.png` | Real-only vs 最佳增强 |
| `figures/v4_best_vs_real_mean_std.png` | 最佳-基线误差棒 |
| `figures/v4_hgb_ap_heatmap_mean.png` | HGB 的 AP 热力图 |

**未版本化的可再生产物**（见 `.gitignore`）：

- `raw_*.npz`（216 个，83 MB）：各配置生成窗口，含 `source_*` 溯源
- `generated_metadata_*.csv`（216 个，207 MB）：逐样本生成元数据
- `figures/generated_data/`（348 MB）：25 设备网格图
- `annotated_device_points_v4.csv`、`window_metadata_v4.json`：可由 `05_scripts/` 重新生成

复现命令见仓库根 `README.md` 与 `06_reports/`。

## 5. 与 v4.2 的选择建议

| 目标 | 推荐 |
|---|---|
| 论文中的增强效果论证 | **v4-evi**（本目录） |
| 异常判定的方法学严谨性 | v4.2（基线经泄漏修复、可复现性修复） |

v4.2 的环境基线确实更准（R² 0.536 vs 0.511），但这**没有**转化为 SOTA 指标的全面优势：
它在 4:6 HGB 下降 0.046、5:5 RF 下降 0.053。因此不能因为基线更先进就宣称整体更优。

两者用途不同，不应混为一谈。

## 6. 局限

1. 标签是统计候选（pseudo anomaly），不是检修确认故障，AP 0.12~0.28 不能当作现场精度。
2. 生成值仍有 `Measure <= 0` 的情况（约 10~11%），物理正值约束未满足。
3. 环境/湿度用于条件化与审计，不代表生成器学到了因果。
4. 窗口固定为点数，不是固定时长。
5. 2 台设备组（2 组）无法归因到具体设备，结果与 3 台组混在同一统计中。
6. 方法排名受种子波动影响，单次运行不足以定论。

# v4 成果总集（device_data_110_220）

本文件夹汇总 v4 异常判定方法及其驱动的论文代码 SOTA 数据增强实验的全部最新成果。
数据源：`device_data_110_220_final.xlsx`（25 设备、16,985 有效点）。
整理时间：2026-10-09。原始运行目录见文末「来源」。

## 目录结构

```
v4_results_20261009/
├── README.md                          # 本文件（总索引）
├── 01_anomaly_labeling/               # v4 判异常方法与标注结果
├── 02_benchmark_base_features/        # SOTA 增强（base 5 维特征）
├── 03_benchmark_v4_evidence_features/ # 【推荐方案】SOTA 增强（v4-evidence 5 维特征）
├── 04_smoke/                          # 小规模冒烟（base / v4evi）
├── 05_scripts/                        # 复现脚本
├── 06_reports/                        # 方法、实验、合并报告与生成数据图索引
├── 07_generated_v4_recheck/            # 生成Measure的固定真实背景v4条件复判
├── 08_v42_baseline_comparison/        # v4.2 环境基线对比（Huber/NLinear/archived/naive）
└── 09_sota_v42/                       # v4.2 标签驱动的 SOTA 增强实验
```

> **当前推荐方案：`03_benchmark_v4_evidence_features/`（v4-evi）。**
> 它是三套候选中唯一在 4 个训练比例、2 种分类器上全部获得正增强收益的配置，
> 详见该目录的 `README_V4EVI.md`。
> `09_sota_v42/` 记录了 v4.2 标签的对照实验；v4.2 基线更准（R² 0.536 vs 0.511），
> 但 SOTA 指标未全面超越 v4-evi，两者用途不同。

> **合并报告**：`06_reports/V4_CONSOLIDATED_REPORT.md` 将 v4 判定方法与两套 SOTA 增强实验合并为一份完整报告（含全部表格、图、复现命令）。

## 一、v4 异常判定方法（01_anomaly_labeling）

v4 沿用 v3 判定原则（时间归一化 + 设备级环境残差 + 湿度分层标准化 + 一致性 + 跳变 + 取消人为保底），
只把无重叠窗口从 64 点缩小到 **32 点**，以捕捉易被 64 点窗口稀释的短事件。

| 指标 | 数值 |
|---|---:|
| 窗口总数 | 547（full 524 + tail 23） |
| 候选异常窗口 | 38（6.95%） |
| `peer-delta jump (per-device)` | 31 |
| `isolated extreme peer jump` | 4 |
| `environment residual and peer agree` | 3 |
| 点级 top5% 高湿占比（原始 peer 差 → 分层 peer z） | 97.5% → 70.9%（基准 70.3%） |

判据（任一成立即候选）：① 环境残差与 peer 偏离同向且持续；② 窗口内 ≥2 次不同步 peer 跳变；
③ 单点孤立极端 peer 跳变（z≥12）；④ 无 peer 设备的强残差。逐点覆盖 25 台全部 0 丢点。

- 方法报告：`06_reports/V4_ANOMALY_LABELING_METHOD.md`
- 结果：`01_anomaly_labeling/`（`protocol_v4.json`、`device_nonoverlap_windows_v4.csv/xlsx`、`device_points_v4.csv`、两张审计图）

## 二、论文代码 SOTA 数据增强（v4 标签驱动）

协议：长度 4 滑动窗口、5 维特征、18 个论文扩散/生成方法（5 个固定上游版 + 13 个机制移植）、
train:test = 2:8 / 3:7 / 4:6 / 5:5、下游 RF/HGB、60 epoch / 20 diffusion steps / 种子 42,123,2024。
标签映射：点落在 v4 候选窗口内即异常，4 点窗口含任一异常点即正类。

### base 特征（`Measure, env_residual, Hum, Temp, delta_measure`）—— 02

HGB，每比例最佳方法：

| 训练:测试 | 最佳方法 | AP 均值±标准差 | Real-only AP | Δ AP |
|---|---|---:|---:|---:|
| 2:8 | IMDiffusion | 0.1548 ± 0.0092 | 0.1217 | +0.033 |
| 3:7 | CCDM | 0.2012 ± 0.0024 | 0.1716 | +0.030 |
| 4:6 | D3R | 0.2020 ± 0.0072 | 0.1764 | +0.026 |
| 5:5 | Diffusion-TS | 0.1988 ± 0.0068 | 0.2149 | −0.016 |

问题：v4 候选以 peer 跳变为主，而 base 特征看不到 peer 信息，Real-only AP 偏低。

### v4-evidence 特征（`Measure, env_residual_z, peer_delta_z, peer_delta_jump_z, delta_measure`）—— 03

把 v4 判定的点级证据直接作为通道（`peer_delta_jump_z` 裁剪到 20，无 peer 匹配处补 0）：

| 训练:测试 | 最佳方法 | AP 均值±标准差 | Real-only AP | Δ AP |
|---|---|---:|---:|---:|
| 2:8 | CoSTI | 0.2298 ± 0.0154 | 0.2062 | +0.024 |
| 3:7 | MTSCI | 0.2208 ± 0.0085 | 0.2071 | +0.014 |
| 4:6 | IMDiffusion | 0.2786 ± 0.0052 | 0.2746 | +0.004 |
| 5:5 | MTSCI | 0.2639 ± 0.0057 | 0.2323 | +0.032 |

结论：加入 v4-evidence 后 Real-only AP 全面提升（+0.02~+0.10），**四个比例的最佳增强全部超过 Real-only**
（+0.004~+0.032），并修掉了 base 版 5:5 增强略低于 Real-only 的问题。

- base 报告：`06_reports/BENCHMARK_BASE_FEATURES_REPORT.md`
- v4-evidence 报告：`06_reports/BENCHMARK_V4EVI_FEATURES_REPORT.md`
- 对比报告：`06_reports/COMPARE_BASE_VS_V4EVI.md`

## 三、复现命令（05_scripts）

```bash
PY=/home/test/miniconda3/envs/jyz_paper/bin/python
# 1) v4 标注
$PY process_device_nonoverlap_windows_v4.py \
    --output runs/device_data_110_220_nonoverlap_windows_v4_20260928 \
    --window-size 32 --peer-group-mode intra-string
# 2) base 特征 SOTA 增强
$PY device_data_paper_augmentation_v4.py --output runs/device_data_110_220_paper_aug_v4_20260930 \
    --feature-set base --epochs 60 --steps 20 --seeds 42,123,2024
# 3) v4-evidence 特征 SOTA 增强
$PY device_data_paper_augmentation_v4.py --output runs/device_data_110_220_paper_aug_v4evi_20261009 \
    --feature-set v4-evi --epochs 60 --steps 20 --seeds 42,123,2024
# 4) 汇总与对比
$PY summarize_formal_results_v4.py --run runs/device_data_110_220_paper_aug_v4evi_20261009
$PY compare_v4_feature_runs.py --base runs/device_data_110_220_paper_aug_v4_20260930 \
    --v4evi runs/device_data_110_220_paper_aug_v4evi_20261009 \
    --output runs/device_data_110_220_paper_aug_v4evi_20261009/COMPARE_V4FEATURE.md
```

脚本运行环境：`/home/test/miniconda3/envs/jyz_paper`（torch 2.2.2+cu121，RTX 4090）。
脚本之间有相对导入（`device_data_paper_augmentation_v4.py` 依赖 `device_data_paper_augmentation.py`、
`process_device_nonoverlap_windows_v4.py`、`all_papers_extended.py`、`upstream_large_augmentation.py`），
请在 `code_reproduction/experiment_augmentation/` 目录内运行；`05_scripts/` 内是本套 v4 脚本的副本。

## 四、关键文件速查

| 文件 | 说明 |
|---|---|
| `01_anomaly_labeling/protocol_v4.json` | v4 阈值、湿度审计、候选统计 |
| `01_anomaly_labeling/device_nonoverlap_windows_v4.xlsx` | 逐窗口特征 + v4 标签（`window_features`/`summary`） |
| `01_anomaly_labeling/device_points_v4.csv` | 点级证据（`env_residual_z`、`peer_delta_z`、`peer_delta_jump_z`） |
| `02_*/all_results.csv` | base 特征逐次结果（比例×方法×种子×分类器） |
| `02_*/formal_v4_summary_mean_std.csv` | base 特征均值±标准差 |
| `03_*/all_results.csv` | v4-evidence 特征逐次结果 |
| `03_*/formal_v4_summary_mean_std.csv` | v4-evidence 均值±标准差 |
| `03_*/formal_v4_best_methods.csv` | v4-evidence 每比例最佳方法与 Real-only 对比 |
| `*/raw_<ratio>_<method>_seed<seed>.npz` | 各方法生成窗口（含 `source_*` 溯源） |
| `*/figures/` | 最佳-基线误差棒图、方法×比例 AP 热力图 |
| `03_*/figures/generated_data/` | 生成信号图：72 个配置 × 3 张图（18 方法 × 4 比例 × seed42） |
| `06_reports/GENERATED_DATA_VISUALIZATION.md` | 生成数据可视化入口、代表配置与完整图索引 |

### 新增：实际生成数据图（2026-10-10）

已补充最新 **v4-evidence** 实验的全部 18 方法、4 个比例、seed=42，共 **216 张 25 设备网格图**（无需重新训练）：

- `real_vs_generated_by_device.png`：真实与生成 Measure 曲线，蓝/红区分继承的正常/正类标签；本次不抽样，全部窗口均绘制。
- `anomaly_localization_by_device.png`：全部正类生成窗口与真实 v4 候选区间，底部时间条显示来源窗口覆盖。
- `local_window_comparison_by_device.png`：每设备一个独立四点生成窗口与其真实来源窗口对比。

图存于 `03_benchmark_v4_evidence_features/figures/generated_data/<ratio>/<method>_seed42/`。
完整图索引及统计：该目录中的 `GENERATED_DATA_VISUALIZATION.md`、`configuration_summary.csv`、`device_generation_audit.csv`。
复现脚本及测试位于 `05_scripts/plot_v4_generated_data.py`、`05_scripts/test_plot_v4_generated_data.py`。

**解释限制：**横轴是精确的来源/锚点时间，不是新生成时间；独立四点窗口不跨样本连线。
红色标签是生成类别标签，不是重新运行 v4 检测的结果；负值保留，后续无来源覆盖不能解释为无异常。
此前合并报告 PDF 保持原样，本次图表与说明作为独立补充成果。

```bash
/home/test/miniconda3/envs/jyz_paper/bin/python plot_v4_generated_data.py \
  --run runs/device_data_110_220_paper_aug_v4evi_20261009 --seeds 42 --details
```

## 五、重要说明与局限

### 新增：生成数据v4复判

最新v4-evidence的216个文件已做逐样本来源背景复判，详见
`07_generated_v4_recheck/V4_GENERATED_RECHECK.md`。每个四点样本独立替换回真实来源位置，
重新计算Measure对应证据，使用32点窗口和v4原阈值；原始标签与NPZ不变。
这是**固定真实背景和标定参考的条件一致性审计**，不是独立四点完整v4检测，也不是现场准确率。
四点生成证据通道不直接作为检测依据；对照及内部跳变检查用于揭示真实背景的影响。
逐样本判定保存在`07_generated_v4_recheck/rechecked_*.csv.gz`。

复现（在`code_reproduction/experiment_augmentation`目录）：

```bash
/home/test/miniconda3/envs/jyz_paper/bin/python -W error audit_v4_generated_recheck.py
/home/test/miniconda3/envs/jyz_paper/bin/python -W error -m unittest test_audit_v4_generated_recheck -v
/home/test/miniconda3/envs/jyz_paper/bin/python -W error validate_v4_generated_recheck.py
```

1. v4 标签是**统计候选（pseudo anomaly）**，不是检修确认故障，不能作为现场报警性能结论。
2. 窗口仍为固定点数，速率特征已按真实时长归一化，但「按固定时间长度切窗」尚未落地。
3. `measure <= 0` 的生成值仍存在（base 11.18%、v4-evi 10.67%），物理正值约束未满足。
4. 环境/湿度目前用于条件化与审计，不代表生成器真正学到了环境因果关系。

## 六、来源（原始运行目录，未删除）

- `code_reproduction/experiment_augmentation/runs/device_data_110_220_nonoverlap_windows_v4_20260928/`
- `code_reproduction/experiment_augmentation/runs/device_data_110_220_paper_aug_v4_20260930/`
- `code_reproduction/experiment_augmentation/runs/device_data_110_220_paper_aug_v4evi_20261009/`
- `code_reproduction/experiment_augmentation/runs/device_data_110_220_paper_aug_v4_smoke_20260930/`
- `code_reproduction/experiment_augmentation/runs/device_data_110_220_paper_aug_v4evi_smoke_20261009/`

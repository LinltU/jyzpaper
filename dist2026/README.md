# dist2026 算法与运行目录

## 当前生产链路

GUI 入口为 `production/main.py`，在线预测器为 `production/humidity_stage_ensemble_predictor.py`。GUI 默认读取：

```text
humidity_stage_feature_ensemble_evaluation/ensemble_manifest.json
├── ../humidity_stage_fmlp_clean_elapsed_evaluation/humidity_stage_fmlp.pth
└── ../humidity_stage_fmlp_v2_high_expert_evaluation/humidity_stage_fmlp.pth
```

目标湿度低于 80% 时使用 clean-elapsed 基础模型；达到或超过 80% 时使用动态高湿专家。两者都是“共享 FMLP + 四阶段残差专家 + P10/P50/P90 分位数输出”，且 Battery 不进入模型。

GUI 的配置 `config/config.ini`、抓取结果 `output/`、旧逐设备模型 `model_output/` 和当前集成模型均由 `production/main.py` 所在位置确定，不受 PowerShell 当前目录影响。推荐从工程根目录运行：

本地 `config/config.ini` 不纳入 Git。连接所需的 `DEFAULT.BASE_URL`、`username`、`password` 必须自行配置；飞书参数 `table_id`、`app_id`、`app_secret`、`wiki_id` 也从该配置读取。管理员模式需要配置 `DEFAULT.admin_password`，未配置时禁用。源码不包含实际连接凭据。

```powershell
& .\scripts\run_gui.ps1
```

## 目录整理结果

| 路径 | 内容 | 当前运行状态 |
|---|---|---|
| `production/main.py` | GUI 生产入口 | 使用 |
| `production/humidity_stage_ensemble_predictor.py` | 在线集成预测器 | 使用 |
| `model/` | FMLP、湿度阶段专家、DDPM 模型定义 | 使用 |
| `data_fetcher/` | 现场数据抓取和位置映射 | 使用 |
| `core/` | 共享特征工程与路径工具 | 使用 |
| `experiments/` | 当前训练、评估、模型选择入口 | 使用 |
| `tests/` | 当前单元测试和 GUI 集成测试 | 使用 |
| `tools/` | 数据合并、位置补全等辅助工具 | 按需 |
| `docs/` | 优化报告、扩散报告、周报 | 文档 |
| `archive/legacy_code/` | 早期训练和 DLinear 对照脚本 | 归档 |
| `archive/legacy_tests/` | 早期 FMLP/LTSF/Transformer 对照测试 | 归档 |

## 最新重训与扩散候选

| 内容 | 路径 |
|---|---|
| 合并训练集 | `latest_retraining_data_20260716/training_merged.csv` |
| 最新真实留出测试集 | `latest_retraining_data_20260716/test_latest_holdout.csv` |
| 未增强模型 | `latest_retraining_fmlp_20260716/trained_models/ensemble/` |
| 25% DDPM 增强模型 | `latest_retraining_fmlp_20260716/diffusion_ensemble_ratio025/` |
| 条件历史残差 DDPM | `evaluate_history_conditioned_residual_diffusion.py` |
| Battery-free 高湿增强正式入口 | `evaluate_diffusion_augmented_high_expert.py` |
| 重训报告 | `latest_retraining_fmlp_20260716/RETRAINING_REPORT.md` |

在 1,715 条真实在线回放样本上：

| 模型 | RMSE | MAE | P10-P90 覆盖率 |
|---|---:|---:|---:|
| 未增强 | 0.21740 | 0.12227 | 80.12% |
| 25% DDPM 增强 | 0.21496 | 0.12098 | 81.46% |

25% DDPM 候选 RMSE 相对降低约 1.13%，但尚未自动切换 GUI 默认 manifest。条件残差和尾部风险候选降低了部分极高湿误差，整体 RMSE/覆盖率没有稳定超过该候选。

## 数据、特征与模型源码

| 类别 | 文件或目录 |
|---|---|
| 原固定训练/验证/测试 | `data/raw/training.csv`、`data/raw/validation.csv`、`data/raw/test.csv` |
| 因果特征工程 | `core/feature_engineering_v2.py` |
| 四阶段训练 | `experiments/evaluate_humidity_stage_fmlp.py` |
| 基础 FMLP 固定切分 | `experiments/evaluate_fmlp_fixed_split.py` |
| 湿度阶段集成 | `evaluate_feature_stage_ensemble.py` |
| 在线数据集回放 | `evaluate_online_ensemble_dataset.py` |
| 模型定义 | `model/` |
| 单元与集成测试 | `tests/test_*.py` |

训练入口的默认数据和默认输出已锚定到本目录。从其他目录执行 `python C:\...\dist2026\experiments\evaluate_humidity_stage_fmlp.py` 时，仍会正确读取 `data/raw/training.csv` 和 `data/raw/test.csv`。

## 模型 manifest 与可移植性

所有 `ensemble_manifest.json` 中的 checkpoint 均相对于 manifest 自身保存。新训练脚本也用 `project_paths.manifest_relative_path()` 写相对路径。运行时 `resolve_project_path()` 支持 manifest 相对路径、环境变量、`~`，以及旧工程绝对路径失效后的根目录重映射。

移动工程时必须复制相关模型目录结构，而不是只复制 `.pth`。历史 `metrics.json` 和日志里的绝对路径只用于追溯实验，不会被加载器使用。

```powershell
# 核心测试和真实迁移冒烟测试
& ..\scripts\run_tests.ps1

# 当前 tests/ 目录下的完整测试
& ..\scripts\run_tests.ps1 -Full
```

## 文档与归档

- `docs/`：FMLP 优化、扩散增强、异常预警及周报。
- `tools/`：数据整理工具，例如 `merge_excel_with_location.py`。
- `archive/`：历史模型、历史实验、旧训练脚本和旧测试，不参与当前 GUI。
- `LTSF-Linear/`：论文参考实现。
- `model_output/`：旧 GUI 的逐设备模型。
- `output/`：GUI 数据采集结果。

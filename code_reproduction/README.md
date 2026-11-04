# 论文/Git 方案复现目录

本目录保存固定版本的上游仓库、工程数据适配器、独立论文分支、统一运行器和真实运行结果。上游源码位于 `upstream/`，保持原样；针对本工程数据的适配代码位于 `adapters/` 和 `individual_paper_branches/`。

## 目录说明

| 目录/文件 | 用途 |
|---|---|
| `upstream/` | 17 个论文官方或公开 Git 仓库，不在其中修改工程逻辑 |
| `adapters/` | 将绝缘子多设备数据转换为预测、插补和异常检测任务 |
| `individual_paper_branches/` | TimeGrad、DiffPLF、CCDM、TSDE、NsDiff 的独立模型与训练循环，不再通过共享 `mode` 模拟 |
| `local_runs/final/` | 正式复现模型、预测表和指标 |
| `_intermediate_archive/` | 历史试跑和缓存，不作为正式结果读取 |
| `reports/REPRODUCTION_REPORT.md` | 论文、源码状态、数据差异和结果汇总 |
| `sources_manifest.json` | 仓库 URL、固定 commit 和获取状态 |
| `run_reproductions.py` | 统一运行入口 |
| `runtime/` | 已验证的本机 Python 3.10 CPU 环境 |

## 运行

推荐从工程根目录执行统一脚本，当前目录可以是任意位置：

```powershell
& "C:\Project\JYZmentor\scripts\run_reproductions.ps1" `
  --output local_runs/final `
  --epochs 40 `
  --imputation-epochs 6 `
  --batch-size 512 `
  --hidden-dim 128 `
  --diffusion-steps 20 `
  --samples 32
```

也可以在本目录直接使用 `runtime/python.exe` 执行 `run_reproductions.py` 和 `generate_reproduction_report.py`。`--only prediction`、`--only imputation`、`--only anomaly` 可只运行一类任务。

默认数据由脚本自身定位到 `../dist2026/latest_retraining_data_20260716/` 下的 `training_merged.csv` 和 `test_latest_holdout.csv`。Battery 不进入任何模型特征。预测使用 24 步因果历史窗口和目标环境/位置条件；插补只对 `Measure/Temp/Hum` 构造人工缺失；异常检测使用正常训练窗口和合成异常压力测试。

## 环境迁移

`runtime/` 约占本复现目录的大部分空间，并带有 Conda 环境的本机前缀记录。同一台机器整体移动时，统一脚本会先测试它是否仍可用；换到另一台机器时，可不复制 `runtime/`，在目标机器创建 Python 3.10 环境并安装 `requirements-reproduction.txt`。

## 复现口径

论文仓库通常面向 ETT、Weather、交通、NASA、SWaT 或临床数据，而本工程是 25 个绝缘子设备、温湿度/位置条件下的测量电压数据。因此本地结果分为两类：

- Treeffuser：使用官方实现完成训练和采样。
- Diffusion-TS：有独立完整分支，包含 Transformer、TrendBlock 和 Fourier 损失。
- TimeGrad、DiffPLF、CCDM、TSDE、NsDiff：保留固定 commit 的官方源码，并在 `individual_paper_branches/` 中分别实现可迁移核心；不是逐层等价重写，也不将不同数据集上的论文指标直接比较。

TimeGrad 的 `pts` 包已通过导入验证；独立分支实验使用本地 `individual_paper_branches/` 的专用训练循环。每个方案是否真正调用官方包、是否只完成机制适配，均记录在运行结果的 `metrics.json` 和复现报告中。

独立分支的最新结果与论文数据集差异说明见：
`reports/INDEPENDENT_PAPER_BRANCHES_20260907.md`。

插补与异常检测方案、对应论文 Git 仓库、本地模型骨架和结果路径见：
`reports/IMPUTATION_ANOMALY_METHOD_MAP_20260907.md`。

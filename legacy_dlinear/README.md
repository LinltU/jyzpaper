# 本地 DLinear 代码归档

上传日期：2026-11-04。来源是用户指定的本地 `code/dist`，保留现有算法，供审阅、版本管理和后续修正。**本目录不是已完成防泄漏修正的 v5 基线。**

## 文件

- `train.py`：原 `DeviceModelTrainer`，含 DLinear/Transformer 训练、模型保存与加载、`predict()`。
- `model/ltsf_nlinear.py`：原 `moving_avg`、`series_decomp`、`NLinearModel`、`DecompositionNLinear`。
- `model/Transformer.py`：原训练模块顶层导入的必要依赖，随附以便导入训练代码。
- `data_fetcher/config_util.py`：只包含本地配置读取类，不含实际配置值或采集服务代码。
- `run.py`：新增命令行入口，默认使用 DLinear，可训练或加载已有模型预测。
- `SOURCE_MANIFEST.json`：原文件 SHA256 与归档文件 SHA256，标明改动。

原算法文件只改了一处：生成本地加密密钥时不再打印密钥本身。模型公式、训练特征、随机切分、损失及预测后处理均保持原样。原目录文件未被修改。

不包含 GUI `main.py`、远程数据采集/API、实际 INI、`license`、原始 XLSX、模型权重和模型序列化文件；本目录是算法核心及其导入依赖，不能作为完整桌面应用启动。

## 使用

请使用单独 Python 环境，勿与仓库根目录的 v4 固定依赖直接混装。依赖清单是从 imports 整理的，未声称恢复原始完整环境；原版本使用旧 pandas 接口。

```powershell
python -m pip install -r legacy_dlinear/requirements.txt
python legacy_dlinear/run.py --help
```

自行在忽略目录中准备 `private/devices.ini`：

```ini
[QUERY]
devices = ["000000000000001"]
```

示例设备 ID 为虚构值，需改成自己的实际工作表名称。数据目录中每个 XLSX 按设备分工作表，代码自动选择该目录最后修改的 XLSX；必须先确认目录中就是要训练的那份文件。列至少包含 `Timestamp, Measure, Temp, Hum`。

```powershell
python legacy_dlinear/run.py --config private/devices.ini --data-folder private/input --device-index 0 train --save private/models/example_dlinear.pkl
python legacy_dlinear/run.py --config private/devices.ini --device-index 0 predict --model private/models/example_dlinear.pkl --temp 25 --hum 60 --history 0.1 0.12 0.11
```

训练和保存仅在显式调用 `train` 时发生。原保存逻辑会生成配套权重及本目录的本地 `license`，应留在私有存储并自行保管；Git 已忽略这些文件。加载旧加密权重需要原密钥，仓库不提供密钥。只加载可信的本地模型文件。

## 当前实现与 v5 的区别

该代码有已识别但本次上传未修改的问题：

- 当前 Measure 的差分与 lag 同时输入，可直接还原目标，移动均值也包含当前值。
- 全量拟合缩放器、随机划分、复用测试集选模型，不能作为严格时间外评估。
- 分解前形状为 `[B,1,C]`，真实历史时间长度为 1，与 v5 要求的 `[B,L,C]` 不同。
- 训练过滤 `Hum≥80`；推理存在高湿混合及经验健康百分比/置信度，不能作为已验证故障概率。

下一步修正要求见 `../prompts/v5_dlinear_anomaly_prompt.md`。本次没有训练模型、执行 v5 标签生成或声称修正已完成。

本次核验范围为语法编译、训练模块导入、模型前向输出形状/有限性、命令行帮助与来源哈希。没有执行完整训练或验证原加密模型兼容性。

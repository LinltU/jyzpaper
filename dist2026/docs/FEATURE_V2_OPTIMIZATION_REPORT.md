# FMLP 特征与阶段集成优化报告

## 目标

本轮优化在不使用设备电量 `Battery` 的前提下，修正安装位置语义，并提升四阶段 FMLP 的整体和极高湿预测性能。原始 `training.csv` 与 `test.csv` 未被修改。

## 实现内容

1. 新增 `feature_engineering_v2.py`：
   - 正确解析 110/220kV、左右侧、上下相、大小号侧、串号、片数和测量片段。
   - 将 6 条 `Hum=0` 记录标记为异常，并按设备进行因果前向填充。
   - 增加按时间归一化的变化率、EMA、MAD、高湿持续时间等动态特征。
   - 增加 `DeviceElapsedDays`，替代 Battery 所隐含的设备运行时间信息。
2. 扩展 `HumidityStageFMLP`：
   - 支持设备嵌入。
   - 支持目标时刻已知的温度、湿度、湿度变化和采样间隔协变量。
3. 新增阶段集成：
   - `Hum < 80` 使用 clean + DeviceElapsedDays 模型。
   - `Hum >= 80` 使用动态特征高湿专家。
   - 两个模型均不使用 Battery。

## 消融结果

所有实验使用相同训练/验证/测试窗口、`seq_len=24`、随机种子 42。

| 方案 | Battery | 特征数 | 整体 RMSE | 极高湿 RMSE |
|---|---:|---:|---:|---:|
| 原四阶段基线 | 是 | 46 | 0.09216 | 0.28027 |
| 原特征仅删除 Battery | 否 | 45 | 0.11059 | 0.32828 |
| 修正位置并保留 Battery | 是 | 46 | 0.09636 | 0.29225 |
| Clean + DeviceElapsedDays | 否 | 46 | 0.09504 | 0.29553 |
| 动态高湿专家 | 否 | 28 | 0.11006 | 0.26177 |
| 阶段集成 | 否 | 双模型 | **0.08696** | **0.26177** |

阶段集成相对原基线：

- 整体 RMSE 改善 5.64%。
- 整体 MAE 从 0.04750 降至 0.04531，改善 4.61%。
- 极高湿 RMSE 改善约 6.60%。
- P10-P90 总体覆盖率为 93.13%。
- 极高湿覆盖率从 76% 提升至 88%，区间相应变宽。

## 关键结论

`Battery` 在旧模型中虽然能提高测试精度，但数据分析显示它与设备运行时间存在明显相关性：各设备 Battery 与运行天数相关系数的中位数约为 -0.63。因此它主要承担了时间代理变量作用，而不是绝缘子测量电压的物理驱动。

使用显式 `DeviceElapsedDays` 后，大部分精度能够恢复；再按湿度阶段选择 clean 模型和动态专家，可以在完全移除 Battery 的同时超过原基线。

当前集成适用于目标时刻温度和湿度已经采集、需要估计测量电压的场景。如果任务改为提前预测下一时刻，必须改用因果路由；现有数据上的因果路由 RMSE 约为 0.20，说明还需要气象预测值或更密集的连续序列。

## 使用方式

一键重新训练并生成集成结果：

```powershell
C:\ProgramData\miniconda3\envs\JYZmentor\python.exe run_feature_optimization.py
```

已验证结果位于：

```text
humidity_stage_fmlp_clean_elapsed_evaluation/
humidity_stage_fmlp_v2_high_expert_evaluation/
humidity_stage_feature_ensemble_evaluation/
```

## 在线 Predictor 与 GUI 接入

新增 `humidity_stage_ensemble_predictor.py`，从
`humidity_stage_feature_ensemble_evaluation/ensemble_manifest.json` 加载普通湿度模型和高湿专家。
在线输入会按训练时保存的特征列、缩放器、湿度填充值、采样间隔和设备起始时间重建，避免训练与部署预处理不一致。

在线预测流程如下：

1. 数据源请求最近 37 条记录，预测器按时间去重和排序，并排除目标时刻记录。
2. 使用 24 条目标时刻之前的历史记录构造 FMLP 窗口；动态高湿特征还需要额外预热记录。
3. `Hum < 80` 路由到 `clean_elapsed`，`Hum >= 80` 路由到 `dynamic_high_expert`。
4. 输出 `P10/P50/P90`，用实测 `Measure` 是否落在区间内进行异常评估，再转换为旧 GUI 使用的六元组。

`main.py` 的设备详情页和主刷新流程均已接入。阶段集成模型优先；当 manifest 不存在、模型加载失败、历史不足或推理异常时，继续按原顺序尝试 Transformer/DLinear 旧模型。旧模型回退仍只接收最近 2 条测量值，不受新模型 37 条历史请求影响。

`Battery` 只保留在设备详情的电量显示和设备健康监控中，未传入在线 predictor，也不在两个 FMLP checkpoint 的特征列中。

## 部署验证

验证包含：

- Python 编译检查：`main.py`、在线 predictor 和旧 DLinear 兼容模块。
- `training.csv + test.csv` 离线回放：普通湿度与极高湿样本的 `P10/P50/P90` 与离线评估逐位一致。
- 无网络 GUI 数据流测试：覆盖普通湿度路由、高湿路由、37 条历史请求、表格更新和集成失败后的旧模型回退。
- 完整特征、模型、阶段集成、在线 predictor 和扩散增强单元测试。

补回了发行目录缺失的 `model/ltsf_nlinear.py` 源文件。该实现与工程 `SuzhouV1.1` 目录中旧 DLinear checkpoint 使用的层结构和参数接口一致；缺失时 GUI 会在导入 `train.py` 阶段直接退出。

在线真实设备接口尚未在本地调用。当前验证使用 CSV 回放模拟
`get_last_n_data()` 的 `(Measure, Temp, Hum, Timestamp, Battery, Version)` 返回格式。正式部署时还应观察历史接口是否稳定返回至少 25 条有效、时间戳不同且早于目标时刻的记录；高湿动态特征建议返回完整 37 条。

旧 Transformer/DLinear 回退分支的控制流已经通过模拟测试，但当前目录中的部分旧 `.pkl` 是由 `scikit-learn 1.6.1 + NumPy 2.x` 保存，现有 GUI 环境为 Python 3.8、`scikit-learn 0.24.0 + NumPy 1.23.5`，无法可靠反序列化。若部署要求实际启用旧模型回退，应在当前 Python 3.8 环境重新导出旧模型，或迁移到 Python 3.9 以上并安装与产物一致的依赖后重新验证。新 FMLP checkpoint 使用 PyTorch 状态字典和 JSON manifest，不受此 `.pkl` 版本问题影响。

当前路由假设目标时刻的温湿度已经采集。如果业务改为提前预测未来时刻，不能继续使用目标湿度路由，应提供气象预报值或切换到只依赖历史湿度的因果路由并重新训练、评估。

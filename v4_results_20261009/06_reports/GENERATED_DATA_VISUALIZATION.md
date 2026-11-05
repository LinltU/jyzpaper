# v4 生成数据可视化补充成果

绘制对象：最新 **v4-evidence** 正式实验，18 个方法 × 4 个训练:测试比例 × seed=42，合计 **72 个配置、216 张图**；每张图包含 25 个设备面板。

## 完整索引与统计

- [完整说明及 72 配置图索引](../03_benchmark_v4_evidence_features/figures/generated_data/GENERATED_DATA_VISUALIZATION.md)
- [配置级统计](../03_benchmark_v4_evidence_features/figures/generated_data/configuration_summary.csv)
- [设备级覆盖与负值审计](../03_benchmark_v4_evidence_features/figures/generated_data/device_generation_audit.csv)
- [绘图协议与软件版本](../03_benchmark_v4_evidence_features/figures/generated_data/plot_protocol.json)

每配置的三张图：

1. **真实与生成 Measure 曲线对比**：全部独立四点窗口，蓝色为正常类，红色为正类，灰色为真实曲线。
2. **正类生成窗口定位**：全部正类生成值；浅红背景为真实 v4 候选区间，底部蓝/红时间条分别显示全部正常/正类锚点覆盖。
3. **局部四点对比**：每设备优先取正类，否则取正常类；来源 ID 排序取中间样本，不挑选“最好看”的生成样本。

## 推荐浏览入口

以下配置参考三种子 HGB AP 汇总结果，但图仅展示 seed=42，不能据此推出整体生成质量最佳；2:8 的 CoSTI 与 SSSD、WaveStitch 存在 AP 并列。

| 比例 | 方法 | 曲线对比 | 异常定位 | 局部四点 |
|---|---|---|---|---|
| 2:8 | CoSTI | [图](../03_benchmark_v4_evidence_features/figures/generated_data/2to8/CoSTI_seed42/real_vs_generated_by_device.png) | [图](../03_benchmark_v4_evidence_features/figures/generated_data/2to8/CoSTI_seed42/anomaly_localization_by_device.png) | [图](../03_benchmark_v4_evidence_features/figures/generated_data/2to8/CoSTI_seed42/local_window_comparison_by_device.png) |
| 3:7 | MTSCI | [图](../03_benchmark_v4_evidence_features/figures/generated_data/3to7/MTSCI_seed42/real_vs_generated_by_device.png) | [图](../03_benchmark_v4_evidence_features/figures/generated_data/3to7/MTSCI_seed42/anomaly_localization_by_device.png) | [图](../03_benchmark_v4_evidence_features/figures/generated_data/3to7/MTSCI_seed42/local_window_comparison_by_device.png) |
| 4:6 | IMDiffusion | [图](../03_benchmark_v4_evidence_features/figures/generated_data/4to6/IMDiffusion_seed42/real_vs_generated_by_device.png) | [图](../03_benchmark_v4_evidence_features/figures/generated_data/4to6/IMDiffusion_seed42/anomaly_localization_by_device.png) | [图](../03_benchmark_v4_evidence_features/figures/generated_data/4to6/IMDiffusion_seed42/local_window_comparison_by_device.png) |
| 5:5 | MTSCI | [图](../03_benchmark_v4_evidence_features/figures/generated_data/5to5/MTSCI_seed42/real_vs_generated_by_device.png) | [图](../03_benchmark_v4_evidence_features/figures/generated_data/5to5/MTSCI_seed42/anomaly_localization_by_device.png) | [图](../03_benchmark_v4_evidence_features/figures/generated_data/5to5/MTSCI_seed42/local_window_comparison_by_device.png) |

## 解释限制

- 横轴为精确的四个**来源时间**，不是新生成的时间；独立窗口不跨样本连线。
- 生成“异常”继承生成类别，**没有重新执行 v4 peer 判定**，不能当作独立检测结果。
- 来源设备是锚点设备，peer_group 生成不保证真实设备身份被重现。
- 生成仅覆盖 fit 来源窗口；之后没有生成覆盖，不表示之后没有异常。
- 所有逆缩放值保留，包括负数和零；本次对比图没有抽样或裁剪。
- 部分生成器会保留来源观测上下文，因此不是四个点都一定为新生成点。

复现和测试脚本见 `../05_scripts/`。本次不重新训练、不修改原始 NPZ、不改变此前合并报告及 PDF。

# DiffAD 相关论文与公开代码方案

## 1. DiffAD 基准

| 项目 | 内容 |
|---|---|
| 论文 | Chunjing Xiao et al., *Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models*, KDD 2023 |
| 官方代码 | https://github.com/ChunjingXiao/DiffAD |
| 仓库信号 | GitHub 页面显示约 125 stars、17 forks，Python 项目，包含 `time_train.py`、`time_test.py`、`config/`、`model/` 和 `tf_dataset/` |
| 核心机制 | 对被遮挡的时间序列进行条件扩散插补；用插补误差和 weight-incremental 机制形成异常分数 |
| 官方数据 | PSM、MSL、SMAP、SMD、SWaT |
| 本工程适配 | 只用 `error=0` 稳定正常双设备窗口训练；遮挡 `[M4,M7]` 或残差，扩散恢复后计算重构误差 |
| 边界 | 当前本地 `DiffAD` 结果是机制适配，不是官方 SMAP/PSM 配置原样运行 |

## 2. 优先复现的高相关方案

| 论文/方案 | 公开代码 | 主要思想 | 对当前数据的适配度 |
|---|---|---|---|
| D3R, *Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection*, NeurIPS 2023 | https://github.com/ForestsKing/D3R | 动态分解、去漂移、扩散重构；减少趋势漂移造成的误报 | 高。适合重新上电衰减和跨阶段漂移，但应先按实验日期校准 |
| CSDI, *Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation*, NeurIPS 2021 | https://github.com/ermongroup/CSDI | 条件 score-based diffusion，随机掩码后插补 | 中。适合插补/重构异常评分，不是首选稳定数据生成器 |
| TimeGrad, *Diffusion-based Probabilistic Forecasting with Deep Neural Networks* | https://github.com/zalandoresearch/pytorch-ts | RNN/GRU 条件扩散预测 | 中高。可把正常残差作为预测目标，但当前短窗口和小样本需要降低模型规模 |
| Diffusion-TS, *Diffusion-TS: Interpretable Diffusion for General Time Series Generation* | https://github.com/Hundredl/Diffusion-TS | 趋势/季节分解、Transformer、Fourier 约束 | 中。可做短窗口生成对照，但当前稳定段周期性弱，不能优先于残差 DDPM |
| NsDiff, *Non-stationary Diffusion Models for Time Series Forecasting* | https://github.com/amazon-science/nsdiff | 局部非平稳归一化和条件扩散 | 高。适合不同实验批次稳态水平不同的问题，但应生成残差而非原始 Measure |
| IMDiffusion, *Imputed Diffusion Models for Multivariate Time Series Anomaly Detection* | https://github.com/17000cyh/IMDiffusion | 多次插补/扩散样本的不确定性异常评分 | 中。适合重构误差和不确定性评分，需改为双设备共享环境输入 |
| ProDiffAD, *Progressively Distilled Diffusion Models for Multivariate Time Series Anomaly Detection* | https://github.com/AiShare-WHU/ProDiffAD | 教师扩散模型向快速学生模型蒸馏 | 中。适合部署加速，当前样本规模不足以优先做完整蒸馏 |
| TIMEADDM, *Unsupervised Anomaly Detection for Multivariate Time Series Using Diffusion Model* | https://github.com/Hurongyao/TIMEADDM | 循环表示累积全局时间依赖，再进行扩散重构/评分 | 中。可用于对比，但官方数据和本工程采样过程差异较大 |

## 3. GitHub 搜索到的较新候选

这些仓库公开可见，但 star 数较低或代码成熟度尚未确认，适合调研和小规模试跑，不应替代已有高引用基线。

| 方案 | 代码 | 页面显示特点 | 建议 |
|---|---|---|---|
| TSADC, *Contaminated Multivariate Time-Series Anomaly Detection with Spatio-Temporal Graph Conditional Diffusion Models*, UAI 2025 | https://github.com/Armanfard-Lab/TSADC | 面向污染训练集的时空图条件扩散 | 低到中。当前只有两个空间测量点，可把第4/7片视作两节点，但样本太少，图结构收益待验证 |
| ICDiffAD, *Implicit Conditioning Diffusion Model for Time Series Anomaly Detection* | https://github.com/ZhangFan21/ICDiffAD | 隐式条件扩散异常检测 | 中。可作为条件残差扩散对照，但需先确认 README、配置和数据接口 |
| CT-DDPM, *Anomaly Detection of Multivariate Time Series with Copula and Transformer-based Denoising Diffusion Probabilistic Models* | https://github.com/4Aver/CT-DDPM | Copula 建模变量相关性 + Transformer DDPM | 高潜力。Copula/相关结构与 M4、M7、M4-M7、M4/M7 的联合关系较匹配，但仓库 star 数较低，应先做 smoke test |
| C3DAD, *Causal-consistent conditional diffusion for multivariate time series anomaly detection* | https://github.com/Tian-siyuan/C3DAD | 因果一致性条件扩散 | 中。可用于区分共享温湿度共因和局部 Measure 偏离，但需要确认代码稳定性 |
| FACD, *Frequency-Aware Time Series Anomaly Detection with Conditional Diffusion Model* | https://github.com/ShenLi0127/FACD | 频域感知条件扩散 | 低到中。当前稳定窗口短且周期性弱，作为对照而不是主模型 |
| DiffusionBased-TSAD | https://github.com/srt-tkyk/DiffusionBased-TSAD | 工业多变量传感器 Conditional DDPM | 低。页面显示为个人/小型实现，适合参考，不作为主要论文证据 |
| DiffusionAD | https://github.com/HodaElmi/DiffusionAD | 潜空间扩散工业时间序列异常检测 | 低到中。需先确认论文、数据和训练脚本完整性 |

## 4. 与本工程的关键数据差异

DiffAD 等公开论文通常使用长时间、多变量、密集采样并带有明确异常标签的数据集，例如 SMAP、MSL、SMD、SWaT、PSM。本工程的 `experiment.xlsx` 有以下差异：

- 只有 3 次实验日期和两个同步设备；
- 设备安装在同一串绝缘子第4片和第7片；
- `error!=0` 表示实验设置了某个零值绝缘子，但最终部署目标只需判断“是否存在零值”；
- 重新断电上电后 Measure 会经历衰减，采集时间未必覆盖完整稳态；
- 第4片和第7片共享环境，温湿度应以同步均值为主；
- 跨实验日期存在稳态 Measure 批次漂移；
- 异常样本数量远少于正常样本。

因此不能直接把官方仓库的窗口、阈值和训练集配置照搬。当前最合理的接口是：

```text
输入：[M4, M7, Temp_shared, Hum_shared, M4-M7, M4/M7, elapsed_min]
训练：只使用后期稳定的 error=0 窗口
目标：正常双设备残差或正常窗口重构
输出：是否偏离正常状态的异常分数
```

## 5. 推荐复现顺序

1. D3R：先处理上电衰减/动态漂移，作为最直接的稳定基线。
2. DiffAD：条件插补重构，适合输出异常分数。
3. CT-DDPM：验证 Copula/Transformer 是否能保留 M4/M7 联合关系。
4. NsDiff 残差版：处理不同日期的稳态尺度变化。
5. TimeGrad 残差版：作为条件概率预测对照。
6. Diffusion-TS 短窗口版：仅作生成对照，不作为首选异常检测器。
7. TSADC/C3DAD/FACD：在前述基线完成后再做小规模探索。

## 6. 官方代码与本地适配口径

- DiffAD、D3R、CSDI、NsDiff、TimeGrad、Diffusion-TS 等上游仓库已保存或登记在 `code reproduction/upstream/` 和 `sources_manifest.json`。
- `code reproduction/adapters/` 与 `experiment_augmentation/` 中的结果是面向本工程数据协议的机制适配。
- 上游代码的官方数据加载器、原论文数据集和完整超参数没有被替换成“本工程原样复现”的证据。
- 新候选仓库在未完成依赖、数据接口和训练 smoke test 前，只能标记为“待复现候选”。

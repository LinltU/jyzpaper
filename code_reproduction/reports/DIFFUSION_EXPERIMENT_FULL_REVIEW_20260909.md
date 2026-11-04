# diffusion 文献、Git 代码与 experiment.xlsx 本地测试报告

> 当前正式复测报告已更新为 [EXPERIMENT_ONLY_DIFFUSION_RETEST_20260909.md](<C:/Project/JYZmentor/code reproduction/reports/EXPERIMENT_ONLY_DIFFUSION_RETEST_20260909.md>)。本文件保留早期调研表和历史方案信息；需要查看本轮只针对 `experiment.xlsx` 的最终指标时，以新报告为准。

## 结论摘要

当前工程的在线目标是二分类：`error=0` 为正常绝缘子串，`error!=0` 为存在零值绝缘子。针对 `experiment.xlsx`，最适合的输入是两台设备的联合稳定状态：

```text
[M4, M7, Temp_shared, Hum_shared, M4-M7, M4/M7, logratio]
```

其中第4片和第7片设备按最近时间戳配对，`Temp_shared=(Temp4+Temp7)/2`，`Hum_shared=(Hum4+Hum7)/2`。本地按实验日期用后期稳定正常窗口校准的 Mahalanobis 方法效果最好；在当前小样本中，20% 正常残差扩散增强没有带来额外提升。

本轮用户指定的测试目标只有 `experiment.xlsx`。其中 `error=0` 是正常绝缘子串，`error!=0` 是实验中实际替换零值绝缘子后产生的有标签异常工况。下面凡是涉及 `training.csv/test.csv` 或人工 6σ 的内容，均不属于本轮结论。

需要严格区分：

- 官方仓库直接运行：本轮完成上游源码固定、编译检查；TimeGrad/Treeffuser 可导入，C3DAD 公开 Lite pipeline 及其 `experiment.xlsx` 双设备输入适配 smoke test 可运行。
- 论文机制本地适配：本报告中的正式数值结果均来自 `experiment.xlsx`，不是把论文官方数据集结果或 `training.csv/test.csv` 压力测试结果改写成本工程实验结果。
- 待复现候选：Zotero 中较新的 CT-DDPM、MAFCD、Graph Diffusion、self-conditioning diffusion 等已完成文献和代码入口登记，但尚未在本地稳定运行。

## A. experiment.xlsx 的数据协议

实验数据有 3 个日期、两个同步设备，分别安装在第4片和第7片绝缘子上。数据按最近时间戳配对，在同一个 `error` 连续阶段内切 6 步窗口，避免混合阶段。

稳定正常专项实验只使用每个日期最早的稳定 `error=0` 窗口进行校准，稳定窗口按 M4/M7 斜率和波动筛选，每个日期取动态性最低的 40%。全部正常窗口 541 个，稳定窗口 217 个。

## B. 针对 experiment.xlsx 的本地测试

| 方案 | 本地代码/来源 | 训练数据 | 本地效果 | 结论 |
|---|---|---|---|---|
| 稳定正常 Mahalanobis | `experiment_augmentation/within_date_stable_benchmark.py` | 每个日期最早稳定 `error=0` | AUROC `0.9969`，AUPRC `0.9675`，平衡准确率 `0.9762`，正常误报率 `0` | 当前首选基线 |
| 稳定正常 + 20% 残差 DDPM | `experiment_augmentation/stable_residual_diffusion.py`、`within_date_stable_benchmark.py` | 只生成正常残差 `[r4,r7,rdiff,rratio]` | AUROC `0.9969`，AUPRC `0.9675`，无增益 | 可作为少量增强，不作为主检测器 |
| D3R 风格去趋势评分 | D3R 机制适配 | 稳定正常窗口 | AUROC `0.8413`，AUPRC `0.4085`，误报率约 `11.2%` | 可作动态变化对照 |
| DiffAD 风格正常重构 | DiffAD 机制适配 | 稳定正常窗口 | AUROC `0.9874`，AUPRC `0.7452`，误报率约 `1.7%` | 次优重构对照 |
| IMDiffusion 风格掩码重构 | IMDiffusion 机制适配 | 稳定正常窗口 | AUROC `0.9772`，AUPRC `0.6307`，误报率约 `3.0%` | 可作不确定性/插补对照 |

实验结果文件：`code reproduction/experiment_augmentation/runs/within_date_stable_20260909_v2/`。上述分数是同一实验日期内稳定校准后的实验室结果，不代表现场部署精度。

## C. 历史主工程预测方案（不纳入本轮 experiment 结论）

主工程使用 24 步窗口、训练 7569、验证 1351、测试 1715，`Battery` 已排除。下表是 `local_runs/final/prediction` 中的统一本地适配结果。

| 方案 | Git 代码 | 本地工作 | RMSE | MAE | R² | P10-P90覆盖率 | 运行口径 |
|---|---|---|---:|---:|---:|---:|---|
| Diffusion-TS | `Y-debug-sys/Diffusion-TS`，约 481 stars | 趋势/状态条件扩散、Transformer/Fourier 等价本地分支 | 0.21542 | 0.12196 | 0.42142 | 67.87% | 本地机制适配 |
| TimeGrad | `zalandoresearch/pytorch-ts`，约 1370 stars | GRU 条件残差 DDPM | 0.23913 | 0.13429 | 0.28705 | 68.92% | 本地预测协议适配，上游包可导入 |
| CCDM | `LSY-Cython/CCDM`，约 7 stars | 通道独立投影与交叉通道融合 | 0.24309 | 0.13709 | 0.26324 | 71.31% | 本地机制适配 |
| NsDiff | `wwy155/NsDiff`，约 65 stars | 局部非平稳归一化和残差扩散 | 0.24432 | 0.13898 | 0.25578 | 71.84% | 本地机制适配 |
| Treeffuser | `blei-lab/treeffuser`，约 57 stars | LightGBM 条件分布 + 扩散采样 | 0.24402 | 0.14743 | 0.25760 | 93.82% | 官方包实际运行 |
| DiffPLF | `LSY-Cython/DiffPLF`，约 41 stars | 局部趋势/尺度和残差扩散 | 0.24949 | 0.14325 | 0.22393 | 71.60% | 本地机制适配 |
| TSDE | `EQTPartners/TSDE`，约 31 stars | 掩码自监督、趋势感知条件扩散 | 0.24949 | 0.14325 | 0.22393 | 71.60% | 本地机制适配 |

主工程预测的点精度候选是 Diffusion-TS；区间覆盖率候选是 Treeffuser。两者目标不同，不能只按单一指标判断。

## D. 历史主工程插补方案（不纳入本轮 experiment 结论）

插补测试是在已有窗口内人工遮挡，不等于真实设备离线时间段补齐。

| 方案 | Git 代码 | 主要工作 | 点缺失 MAE | 连续块 MAE | 运行口径 |
|---|---|---|---:|---:|---|
| CoSTI | `javiersgjavi/CoSTI`，约 7 stars | 一致性、状态偏置和掩码扩散插补 | 0.4380 | 0.6860 | 本地机制适配 |
| CSDI | `ermongroup/CSDI`，约 461 stars | 条件 score-based diffusion 插补 | 0.8811 | 0.6969 | 本地机制适配，官方代码已固定 |
| MTSCI | `JeremyChou28/MTSCI`，约 33 stars | 多变量条件一致性扩散插补 | 0.5903 | 0.5507 | 本地机制适配 |
| SSSD | `ai4healthuol/sssd`，约 335 stars | 结构化状态空间扩散插补 | 0.5553 | 0.6209 | 本地机制适配 |
| SSD-TS | `decisionintelligence/SSD-TS`，约 8 stars | 结构化扩散时间序列插补 | 0.5996 | 0.6413 | 本地机制适配 |
| TSDE-Imputation | `EQTPartners/TSDE`，约 31 stars | 掩码/插值/预测统一自监督 | 0.5934 | 0.6230 | 本地机制适配 |

当前插补点缺失场景中 CoSTI 最好；连续块缺失中 MTSCI 最好。由于缺失是人工构造，不能把该结果直接解释为现场离线插补能力。

## E. 历史主工程异常检测参考（不纳入本轮 experiment 结论）

本节是此前针对主工程 `training.csv/test.csv` 的历史参考结果，不再作为本轮 `experiment.xlsx` 方案比较依据。该数据使用人工 6σ Measure level-shift 压力测试，因此表中的“伪异常”与 `experiment.xlsx` 的 `error!=0` 实验异常不是同一概念。

需要与 `experiment.xlsx` 区分：`experiment.xlsx` 是实验室受控实验，`error=0` 表示正常绝缘子串，`error!=0` 表示实验中人为设置了零值绝缘子，属于**有标签的实验室异常**。该数据在本报告的“稳定段异常检测”章节单独评估，使用的是真实实验标签，不是人工 6σ 伪异常。

| 方案 | Git 代码 | 主要工作 | 伪异常 AUROC | AUPRC | 6σ检测率 | 正常误报率 | 运行口径 |
|---|---|---|---:|---:|---:|---:|---|
| DiffAD | `ChunjingXiao/DiffAD`，约 125 stars | 条件扩散插补 + weight-incremental 评分 | 0.9979 | 0.9869 | 95.51% | 0.52% | 主工程 6σ 压力测试，本地机制适配 |
| TIMEADDM | `Hurongyao/TIMEADDM`，约 20 stars | 循环表示累积时间依赖 + 扩散评分 | 0.9977 | 0.9771 | 84.62% | 0.52% | 主工程 6σ 压力测试，本地机制适配 |
| ProDiffAD | `AiShare-WHU/ProDiffAD`，约 2 stars | 扩散教师/学生渐进蒸馏思想 | 0.9913 | 0.9674 | 91.67% | 0.52% | 主工程 6σ 压力测试，本地评分适配，未完成官方蒸馏 |
| IMDiffusion | `17000cyh/IMDiffusion`，约 115 stars | 多次插补不确定性评分 | 0.9836 | 0.9035 | 69.87% | 0.52% | 主工程 6σ 压力测试，本地机制适配 |
| D3R | `ForestsKing/D3R`，约 96 stars | 动态分解、去漂移、扩散重构 | 0.9533 | 0.5824 | 10.26% | 0.52% | 主工程 6σ 压力测试，本地机制适配 |

本节结果仅保留用于追溯历史工程，不参与当前结论。`experiment.xlsx` 应以本报告 B 节的稳定段有标签结果为准。

## E1. experiment.xlsx 有标签异常检测结论

针对 `experiment.xlsx` 的正式评价如下：每个日期只用后期稳定的 `error=0` 窗口训练正常基线；测试使用该日期全部窗口；正类直接采用原始 `error!=0` 标签；目标只判断是否存在零值绝缘子，不判断具体片号。

| 方案 | AUROC | AUPRC | 平衡准确率 | 正常误报率 |
|---|---:|---:|---:|---:|
| 稳定正常 Mahalanobis | 0.9969 | 0.9675 | 0.9762 | 0 |
| DiffAD 风格重构 | 0.9874 | 0.7452 | 0.9675 | 1.73% |
| IMDiffusion 风格重构 | 0.9772 | 0.6307 | 0.9611 | 3.02% |
| D3R 风格去趋势 | 0.8413 | 0.4085 | 0.7632 | 11.17% |
| 稳定正常 + 20% 残差 DDPM | 0.9969 | 0.9675 | 0.9762 | 0 |

这里的 AUROC/AUPRC 正类是真实实验中的 `error!=0` 窗口；正常误报率是真实 `error=0` 窗口被错误报警的比例。因此，这些结果不是 6σ 伪异常压力测试，而是 `experiment.xlsx` 的受控实验标签评估。

## F. Zotero diffusion 目录审视结果

去重后主要条目包括：Diffusion-TS、CSDI、MTSCI、D3R、DiffAD、ImDiffusion、CT-DDPM、MAFCD、Diff-MTS、Graph Diffusion、self-conditioning diffusion、GRD、扩散模型异常检测、Time Series Anomaly Detection using Diffusion-based Models，以及 USAD、OmniAnomaly、MTAD-GAT 等非扩散基线。

其中优先级最高的新增候选是：

| 候选 | 代码入口/状态 | 适配判断 |
|---|---|---|
| CT-DDPM | `https://github.com/4Aver/CT-DDPM`，约 2 stars | 已抓取仓库；当前公开内容只有 README，未提供可执行 Python 训练代码，因此不能声称已完成模型复现。仅保留为论文/代码线索 |
| MAFCD | Zotero DOI `10.1016/j.inffus.2025.102965`，代码入口需从论文作者仓库确认 | 多层、自适应条件扩散，适合稳定段条件异常评分，代码尚未纳入本地上游 |
| Diffusion models with self-conditioning guidance | Zotero DOI `10.1016/j.knosys.2025.114511` | self-conditioning 可降低小样本扩散不稳定，代码尚未纳入本地上游 |
| Diffusion Graph Model | Zotero DOI `10.1145/3701716.3717376` | 第4片和第7片可作为两个节点，但两节点图收益需要实测 |
| GRD | Zotero 中文条目 | GNN + 扩散异常检测，与双设备关系特征有潜在匹配，需先确认公开代码 |
| TSADC | `https://github.com/Armanfard-Lab/TSADC`，约 2 stars | 已固定 commit 并通过 `compileall`；官方入口依赖 `torch_geometric` 和专用 SMD/ICBEB/DODH/TUSZ 数据，未直接运行本工程 |
| ICDiffAD | `https://github.com/ZhangFan21/ICDiffAD`，约 4 stars | 已固定 commit 并通过 `compileall`；官方入口依赖 `rich/hydra` 和 MSL/SMAP/SMD/PSM/SWAT 数据，未直接运行本工程 |
| C3DAD | `https://github.com/Tian-siyuan/C3DAD`，方法预览 | 已通过 `compileall`，公开 Lite pipeline 成功运行；完整因果选择器和扩散训练骨干未公开 |
| FACD | `https://github.com/ShenLi0127/FACD` | GitHub 页面可访问，但本轮下载包不完整，未纳入可复现实验 |

## G. 新增仓库的代码执行审计

| 方案 | 固定版本/来源 | 编译或运行结果 | 是否得到 `experiment.xlsx` 指标 | 边界 |
|---|---|---|---|---|
| CT-DDPM | `4Aver/CT-DDPM`，约 2 stars | 仅 README，无 Python 源码 | 否 | 论文可分析，公开仓库不足以复现 |
| TSADC | `7013d7d2053044acdd3aa980bcf309d48f94de0d` | `compileall` 通过；导入缺少 `torch_geometric` | 否 | 专用图数据协议 |
| ICDiffAD | `093219ea3f2fe9d3a0656c313eae7b0abf549860` | `compileall` 通过；导入缺少 `rich` | 否 | 专用基准数据和 Hydra 配置 |
| C3DAD | `0cc5d514322853a8230494d528c3472852ad09eb` | `compileall` 通过；公开 Lite pipeline 成功，并用 250626 第4/7片配对数据完成输入适配 smoke test | 否 | 完整 causal selector、diffusion backbone 未公开 |
| FACD | `ShenLi0127/FACD` | 下载包不完整 | 否 | 不引用官方指标 |

“已测试”分为三档：官方仓库实际运行、源码编译通过但未运行官方训练、本地机制适配并得到本工程指标。`experiment.xlsx` 的数值只来自第三档，不能把“编译成功”理解为已经在本工程训练成功。

## H. 论文方案、代码状态与本地效果总表

| 任务 | 论文/方案 | 公开代码 | 主要工作 | 本地结果/状态 | 适配度 |
|---|---|---|---|---|---|
| 预测 | Diffusion-TS | `Y-debug-sys/Diffusion-TS` | 趋势/季节分解、Transformer 条件扩散、Fourier 约束 | RMSE 0.21542；本地机制适配 | 中 |
| 预测 | TimeGrad | `zalandoresearch/pytorch-ts` | GRU 状态条件概率扩散 | RMSE 0.23913；上游包可导入 | 中高 |
| 预测 | NsDiff | `wwy155/NsDiff` | 局部归一化处理非平稳性 | RMSE 0.24432；本地机制适配 | 高 |
| 插补 | CSDI | `ermongroup/CSDI` | 条件 score-based 扩散插补 | 点 MAE 0.8811，块 MAE 0.6969 | 中 |
| 插补 | MTSCI | `JeremyChou28/MTSCI` | 多变量一致性扩散插补 | 点 MAE 0.5903，块 MAE 0.5507 | 中高 |
| 插补 | CoSTI | `javiersgjavi/CoSTI` | 状态偏置与一致性约束插补 | 点 MAE 0.4380，块 MAE 0.6860 | 中 |
| 异常检测 | DiffAD | `ChunjingXiao/DiffAD` | 条件扩散插补重构 + weight-incremental 评分 | 稳定段 AUROC 0.9874 | 高 |
| 异常检测 | D3R | `ForestsKing/D3R` | 动态分解、去漂移、扩散重构 | 稳定段 AUROC 0.8413 | 中高 |
| 异常检测 | IMDiffusion | `17000cyh/IMDiffusion` | 多次插补不确定性评分 | 稳定段 AUROC 0.9772 | 中 |
| 异常检测 | TSADC | `Armanfard-Lab/TSADC` | 污染训练集去污染 + 时空图条件扩散 | 已编译，未得到本地指标 | 中 |
| 异常检测 | ICDiffAD | `ZhangFan21/ICDiffAD` | 隐式条件扩散重构 | 已编译，未得到本地指标 | 中 |
| 异常检测 | C3DAD | `Tian-siyuan/C3DAD` | 因果一致条件、支持图、短/长滞后上下文 | Lite pipeline 可运行，完整扩散未公开 | 中 |
| 异常检测 | CT-DDPM | `4Aver/CT-DDPM` | Copula 相关结构 + Transformer DDPM | 仓库无可执行源码 | 高潜力但不可复现 |

## I. 上游仓库总审计

原先固定上游仓库 17 个；本轮新增 CT-DDPM、TSADC、ICDiffAD、C3DAD、FACD 后，审计清单共 22 个。原 17 个均完成 `py_compile/compileall` 检查；TimeGrad 和 Treeffuser 完成导入检查。新增仓库的编译、导入和 Lite pipeline 结果见上表；其余由于数据集专用入口、依赖或公开内容不完整，记录为“源码编译通过、官方入口未直接运行”。

对应清单：`code reproduction/sources_manifest.json`；官方 smoke 检查：`code reproduction/local_runs/final/official_smoke.json`。

## J. 最终推荐

对 `experiment.xlsx`：

1. 主检测器使用“按实验日期稳定正常校准 + 双设备残差 Robust Mahalanobis”。
2. DiffAD 风格条件插补重构作为第二检测器，并与 Mahalanobis 做分数融合。
3. 只生成少量正常残差，不生成异常片号数据，不把原始 Measure 直接作为扩散生成目标。
4. CT-DDPM、MAFCD、self-conditioning diffusion、Graph Diffusion 仍是论文层面的高相关候选；其中 CT-DDPM 当前公开仓库无可执行源码，不应再标记为“待直接运行”的完整代码方案。
5. 现场部署前需要增加真实挂线异常或更多实验室异常，并按完整实验阶段划分测试集。

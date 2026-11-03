# jyzpaper

真实挂线数据的 v4 异常候选规则重建代码，以及以修正 DLinear 为基线的 v5 实验任务书。

这里的“异常”是统计候选阶段，不是检修确认故障。v4 已实现并核验；v5 当前是待执行的 prompt，不代表已训练或已生成 v5 标签。

## 内容

- `v4/label_v4.py`：时间戳单位已修正的完整 v4 计算与绘图。
- `v4/export_locations.py`：从本地 INI 中安全提取显式设备位置，不继承 DEFAULT。
- `v4/inspect_110kv.py`：独立核验真实时间差、容差、匹配值，并绘制 110kV 同串双设备比较图。
- `v4/build_workbook.mjs`：原交付工作簿构建脚本，可选，依赖 Codex 的 `@oai/artifact-tool` 运行环境。
- `v4/verify_delivery.py`：对原版布局的工作簿、CSV、图像数量及源哈希进行独立核验。
- `tests/test_v4.py`：微秒/纳秒、容差、缺测跳变与配置秘密隔离回归测试。
- `tests/verify_regression.py`：本地私有数据重跑结果与已有时间修正版逐字段比较。
- `prompts/v5_dlinear_anomaly_prompt.md`：可直接交给编程代理执行的中文 v5 prompt。

原始 XLSX、实测 CSV、设备位置映射、INI、模型密钥、模型权重及私人报告不在仓库中。仓库没有声明对这些原始资料的再分发许可。

## 安装与复现

参考环境：Python 3.13.5。使用独立环境安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

将自己的数据放入忽略目录 `private/`。`locations.json` 格式参考 `examples/locations.example.json`，示例 ID 为虚构值，不能直接用于真实数据。

```powershell
python v4/export_locations.py --ini "本地设备配置.ini" --output private/locations.json
python v4/label_v4.py --source private/device_data_110_220_final.xlsx --locations private/locations.json --report "本地报告.pdf" --output outputs/v4
python v4/inspect_110kv.py --output outputs/v4
python -m unittest discover -s tests -v
```

`--report` 可省略：程序只对它计算来源哈希，不解析报告内容。`--skip-plots` 可跳过绘图。所有路径均可由命令行指定。程序保留本数据集的结构断言：25 台设备、9 个同串组；不是任意设备集的通用工具。

源工作簿要求每台设备一个工作表，名称为 15 位设备 ID，包含 `IMEI, Timestamp, Measure, Temp, Hum`。时间保留源文件的本地时钟，不猜测时区。位置字符串须与原配置格式一致，含电压、左右、上下相、大小号侧、串号；当前分组逻辑仅适用于这份数据的单线路配置，跨线路使用前必须增加线路标识，不能仅凭相同方位合并不同线路。

仅需核心计算及图片时，不需要 Node。可选原版工作簿生成需要可用的 Codex `@oai/artifact-tool`：

```powershell
node v4/build_workbook.mjs outputs/v4
python v4/verify_delivery.py --output outputs/v4 --source private/device_data_110_220_final.xlsx
```

普通 Node 环境不保证能安装此依赖。`verify_delivery.py` 针对原始工作簿布局与已知源文件；仅在工作簿生成后运行，不将新的数据集强行匹配到原统计数。

## v4 方法与已知边界

本地未找到报告原始脚本，本仓库依据报告明示流程重建，并公开未明示细节的实现约定，**不声称与报告原实现完全一致**。报告记载 38 个候选窗，本实现得到 27 个；没有向 38 个调参。

1. 每台设备按真实时间排序，去重保留最后一行。32 点不重叠分窗，2–31 点补尾窗，1 点尾部保留为 `-1` 未评估。
2. 逐设备全历史 Huber 拟合 `Measure ~ Temp + Hum + elapsed_days`；这是样本内拟合值，不是独立的未来预测。
3. 残差逐设备中位数/MAD 标准化，再在全设备合并的湿度箱内秩正态变换。湿度边界为 `[0,30,40,50,55,60,65,70,75,80,90,100,101]`。
4. 同串排除自身，各 peer 找最近时间点；容差为目标设备居中 9 点局部中位间隔的 0.5 倍，裁剪至 30 分钟–12 小时，等距选较早，允许复用。不插值。
5. 原始 `peer_delta = 本设备 Measure − 可匹配 peer Measure 中位数`；在同串组×目标湿度箱内秩正态变换。跳变为 `abs(diff(peer_delta))/sqrt(2)/本设备 peer_delta 水平量稳健尺度`。不跨缺测差分；原版不因长间隔额外剔除跳变。
6. 窗口带符号残差均值、peer 标准化均值各取绝对值，在全部窗口池上再做中位数/MAD 标准化，得到窗口证据。
7. A：残差窗口证据≥3、点级 `|residual_z|≥2` 占比≥0.15、peer 窗口证据≥3，两个带符号均值同向。
8. B：同窗至少两个 peer 跳变≥8；C：至少一个≥12。
9. D：没有物理 peer 时，点级残差超限占比≥0.15、绝对残差90分位≥2、残差跳变≥2.5的点至少两个。
10. 任一规则触发则整窗标 1；已评估且未触发为 0；主原因优先顺序 B、C、A、D，全部规则同时保存。排序分数不是概率。

稳健尺度退化时依次使用 IQR/1.349、标准差、1e-12。完整约定、拟合参数和依赖版本写入运行输出 `protocol_v4.json`。

这些拟合、分位映射、窗口池尺度使用全历史，peer 最近匹配和居中间隔也可引用目标时刻之后的观测。因此 v4 是离线回顾性分析，不能拿其表现证明在线检测能力。没有确认健康标签，也不能将 v4 候选当作训练或评估真值。v5 prompt 要求改为时间因果预测和独立校准，并明确报告与 v4 的协议变化。

## 时间单位修正与参考统计

旧代码把 pandas 微秒整数当作纳秒，时间差被低估 1000 倍。现代码在时间搜索前显式转换为 `datetime64[ns]`，并用原始时间相减独立核验每条匹配，不依赖同一整数算法自证。

已知私有源文件 SHA256：

```text
9760253f0a8be4c1fdc461a4ee4c1006e48e605bee8368af508970f55b1ae863
```

时间修正版参考结果（只用于复现核对，不进入标签决策）：

| 指标 | 数值 |
|---|---:|
| 设备 / 同串组 / 有效点 | 25 / 9 / 16985 |
| 窗口 / 完整窗 / 尾窗 | 547 / 524 / 23 |
| 候选窗口 / 候选点 / 未评估点 | 27 / 854 / 2 |
| A / B / C / D 规则触发窗 | 0 / 24 / 6 / 0 |
| peer 尝试 / 接受 / 拒绝 | 29050 / 27527 / 1523 |

B、C 有 3 个重叠窗。时间修正前后有 4 窗、128 点标签改变，候选总数恰好相同，因此不能只检查总数。

110kV 同串两台设备各 58 窗、未触发候选；不是没有做同串比较，也不能据此证明设备健康。两台互比无法单独定位是哪台异常，共同变化可能抵消。

## 看图

`v4_all_devices.png` 和 `device_plots/`：蓝点为原始观测，红点和红线为候选窗口内的观测，浅红底表示窗口起止，深红叉为直接触发的 peer 跳变≥8，灰色空心菱形为未评估点。红线不是数值阈值。所有实际观测相连，长间隔连线不代表中间有数据或做了插值。32 点窗口的真实时长可能不同。

## 复现核验

```powershell
python tests/verify_regression.py --actual outputs/v4 --reference "本地已有时间修正版结果目录"
```

该脚本核对完整点级、窗口级、peer 匹配及设备汇总表，含原始数值、时间戳、缺失位置、证据和标签，避免“候选总数相同但位置错误”。

发布时只检查指定文件，不递归上传原项目、运行时或配置目录。v5 的模型、标签与指标需执行任务书后另行提交。

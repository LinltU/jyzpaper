# v4 标签 SOTA 增强：base 特征 vs v4-evidence 特征（HGB）

两次实验使用完全相同的 v4 标签、比例、方法与种子，唯一区别是 5 维窗口特征：

- base：`Measure, env_residual, Hum, Temp, delta_measure`
- v4-evi：`Measure, env_residual_z, peer_delta_z, peer_delta_jump_z, delta_measure`

| 训练:测试 | base Real-only | v4-evi Real-only | Δ | base 最佳(方法,AP) | v4-evi 最佳(方法,AP) | v4-evi Δ vs 自身 Real-only |
|---|---:|---:|---:|---|---|---:|
| 2to8 | 0.1217 | 0.2062 | +0.0845 | IMDiffusion 0.1548 | SSSD 0.2298 | +0.0237 |
| 3to7 | 0.1716 | 0.2071 | +0.0356 | CCDM 0.2012 | MTSCI 0.2208 | +0.0137 |
| 4to6 | 0.1764 | 0.2746 | +0.0982 | D3R 0.2020 | IMDiffusion 0.2786 | +0.0040 |
| 5to5 | 0.2149 | 0.2323 | +0.0174 | Diffusion-TS 0.1988 | MTSCI 0.2639 | +0.0316 |

## 说明

- Real-only 列：只用真实训练窗口时的 HGB AP（3 个种子均值）。
- 最佳列：每个比例独立取 AP 均值最大的增强方法（描述性，不构成全局最优）。
- v4-evi 把 `env_residual_z`、`peer_delta_z`、`peer_delta_jump_z`（v4 判定的点级证据）直接作为窗口通道，
  使分类器能看到 peer 偏离——这正是 v4 候选（31/38 为 peer 跳变）的主要依据。
- `peer_delta_jump_z` 在 MinMax 前裁剪到 20（单点跳变阈值为 12），避免极端值压平通道。

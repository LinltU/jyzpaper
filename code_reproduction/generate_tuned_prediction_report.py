"""Generate the auditable comparison report for the 2026-09-05 tuning run."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
TUNING = ROOT / "local_runs" / "tuning_all_20260905"
SEEDS = ROOT / "local_runs" / "best_seed_repeats_20260905"
Q90 = ROOT / "local_runs" / "calibration_q090_20260905"
DIST = ROOT.parent / "dist2026"

def f(v, n=4):
    return "-" if pd.isna(v) else f"{float(v):.{n}f}"

def main():
    best = pd.read_csv(TUNING / "best_by_validation.csv")
    seeds = pd.read_csv(SEEDS / "seed_summary.csv")
    q90_rows=[]
    for p in sorted(Q90.rglob("metrics.json")):
        j=json.loads(p.read_text(encoding="utf-8")); t=j["overall"]
        q90_rows.append({"method":j["method"],"rmse":t["rmse"],"mae":t["mae"],"coverage":t["coverage_p10_p90"],"width":t["mean_interval_width"]})
    q90=pd.DataFrame(q90_rows)
    ddpm_path = DIST / "latest_retraining_fmlp_20260716" / "diffusion_online_replay_ratio025" / "metrics.json"
    ddpm=json.loads(ddpm_path.read_text(encoding="utf-8"))
    ddpm_t=ddpm["fmlp"]; ddpm_i=ddpm["interval"]
    baseline_path = DIST / "latest_retraining_fmlp_20260716" / "online_replay" / "metrics.json"
    baseline=json.loads(baseline_path.read_text(encoding="utf-8"))
    base_t=baseline["fmlp"]; base_i=baseline["interval"]

    rows=[]
    for _,r in best.iterrows():
        s=seeds[seeds.method==r.method].iloc[0]
        q=q90[q90.method==r.method].iloc[0]
        rows.append({
            "method":r.method,
            "selection_config":r.config,
            "official_package_used":r.method=="Treeffuser",
            "validation_rmse":r.validation_rmse,
            "test_rmse":r.test_rmse,
            "test_mae":r.test_mae,
            "test_r2":r.test_r2,
            "test_coverage_p10_p90":r.test_coverage_p10_p90,
            "test_interval_width":r.test_mean_interval_width,
            "seed_mean_rmse":s.test_rmse_mean,
            "seed_std_rmse":s.test_rmse_std,
            "q090_coverage":q.coverage,
            "q090_width":q.width,
        })
    comparison=pd.DataFrame(rows)
    comparison.to_csv(ROOT / "results" / "prediction_tuned_comparison_20260905.csv",index=False,encoding="utf-8-sig")

    lines=[]
    lines += ["# 预测论文方案重新调参与对比报告（2026-09-05）", "", "## 1. 结论", ""]
    lines += ["本轮对 7 个公开论文/Git 预测方案使用同一份本工程数据、同一时间切分和同一 24 步因果窗口重新训练，并使用验证集选择参数。测试集只在参数确定后用于一次最终报告。", ""]
    lines += ["调参后，Diffusion-TS 仍是公开论文方案适配组中最优：验证集选定配置的测试 RMSE 为 0.20863，三个随机种子 RMSE 均值为 0.20958，标准差为 0.00093。", ""]
    lines += ["这不是直接加载 Diffusion-TS 预训练模型；官方仓库源码保留在 `upstream/Diffusion-TS`，本地结果是其趋势/状态条件扩散机制在本工程数据上的适配。", ""]
    lines += ["", "## 2. 统一实验条件", "", "- 真实训练记录：9,520 条；真实留出测试记录：1,715 条。", "- 训练窗口：7,569；验证窗口：1,351；历史长度：24。", "- Battery 不进入历史特征、条件特征或任何扩散模型。", "- 目标为下一时刻 Measure，扩散模型生成条件残差分布并输出 P10/P50/P90。", "- 论文方案：TimeGrad、DiffPLF、CCDM、TSDE、NsDiff、Diffusion-TS、Treeffuser。", "- Treeffuser 使用官方包；其余为固定 Git 提交源码基础上的论文核心机制适配。", ""]
    lines += ["", "## 3. 验证集选参后的最终测试对比", "", "| 方法 | 选定配置 | 验证RMSE | 测试RMSE | 测试MAE | R² | P10-P90覆盖率 | 区间宽度 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in comparison.sort_values("test_rmse").iterrows():
        lines.append(f"| {r.method} | {r.selection_config} | {f(r.validation_rmse)} | {f(r.test_rmse)} | {f(r.test_mae)} | {f(r.test_r2)} | {float(r.test_coverage_p10_p90):.2%} | {f(r.test_interval_width)} |")
    lines += ["", "说明：表中配置按验证 RMSE、验证 MAE 选择，不按测试 RMSE 选择。", ""]
    lines += ["", "## 4. 三随机种子稳定性", "", "| 方法 | RMSE均值 | RMSE标准差 | 最小RMSE | 最大RMSE | MAE均值 | 覆盖率均值 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in seeds.sort_values("test_rmse_mean").iterrows():
        lines.append(f"| {r.method} | {f(r.test_rmse_mean)} | {f(r.test_rmse_std)} | {f(r.test_rmse_min)} | {f(r.test_rmse_max)} | {f(r.test_mae_mean)} | {float(r.test_coverage_p10_p90_mean):.2%} |")
    lines += ["", "Diffusion-TS 的随机种子波动最小，且三次结果都低于 0.211；这比单次 0.21542 的旧结果更有参考价值。", ""]
    lines += ["", "## 5. 区间校准敏感性", "", "默认设置使用 0.80 非一致性分位点。针对本工程的异常预警需求，额外测试了 0.90 分位点；它不改变 P50 点预测，只扩大 P10-P90 区间。Diffusion-TS 的覆盖率由 72.48% 提高到 83.03%，平均区间宽度由 0.2257 增加到 0.3156。", "", "该 0.90 结果作为区间校准敏感性实验，不替代点预测主结果；正式部署前应在独立校准集上确定校准分位点。", ""]
    lines += ["", "## 6. 与本工程 DDPM 候选比较", "", "| 模型 | RMSE | MAE | 覆盖率 | 说明 |", "|---|---:|---:|---:|---|", f"| 纯真实FMLP | {f(base_t['rmse'])} | {f(base_t['mae'])} | {base_i['coverage_p10_p90']:.2%} | 工程基线 |", f"| 25% DDPM增强FMLP | {f(ddpm_t['rmse'])} | {f(ddpm_t['mae'])} | {ddpm_i['coverage_p10_p90']:.2%} | 当前工程正式候选 |", f"| 调参 Diffusion-TS | {f(comparison.loc[comparison.method=='Diffusion-TS','test_rmse'].iloc[0])} | {f(comparison.loc[comparison.method=='Diffusion-TS','test_mae'].iloc[0])} | {comparison.loc[comparison.method=='Diffusion-TS','test_coverage_p10_p90'].iloc[0]:.2%} | 公开论文机制适配 |"]
    lines += ["", "调参后 Diffusion-TS 的点预测优于当前 25% DDPM 候选，但默认区间覆盖率仍低于 DDPM。因此：若目标是点预测，优先研究调参后的 Diffusion-TS；若目标是预测+异常预警联合部署，仍需采用独立校准集重新标定区间，不能只凭 RMSE 切换。", ""]
    lines += ["", "## 7. 产物", "", "- 调参脚本：`tune_prediction_reproductions.py`", "- 28 组调参结果：`local_runs/tuning_all_20260905/`", "- 三种子重复：`local_runs/best_seed_repeats_20260905/`", "- 区间校准敏感性：`local_runs/calibration_q090_20260905/`", "- 机器可读对比表：`results/prediction_tuned_comparison_20260905.csv`", "- 每个试验目录均包含 `metrics.json`、`predictions.csv`、`training_history.json` 和运行日志。", ""]
    (ROOT / "reports" / "TUNED_PREDICTION_REPORT_20260905.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(ROOT / "reports" / "TUNED_PREDICTION_REPORT_20260905.md")

if __name__ == "__main__": main()

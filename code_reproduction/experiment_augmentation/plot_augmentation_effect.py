"""增强前后下游效果对比图 (anomaly detection).

固定真实测试集
--------------
对每个 (生成器, 种子, 留出测试日期) 组合, 真实测试窗口的 ``test_id`` 与 ``label``
完全一致 (同一留出日期只切一次测试集), 所以「只用真实数据训练」与「真实+生成数据训练」
两个模型是在**完全相同的测试窗口和真值**上评分并比较的。

重要说明 (不配对求点误差)
--------------------------
新增的生成窗口在真实测试集中**没有一一对应的原始真值**, 因此本脚本绝不把生成窗口与真实
窗口配对来算逐点误差。普通生成增强只比较: ① 测试集上的下游检测收益; ② 生成样本的分布/结构
(后者由 ``plot_data_comparison.py`` 负责)。只有重建/插补这类「输入即目标、共享条件且有对应
目标」的任务才适合逐点比较; 本工程的增强实验不是这种, 故不生成「预测曲线/绝对误差箱线图」,
该分支在报告正文中以说明替代 (见 ``aggregate_all_papers.py`` 第 10 节)。

生成图 (全部在 ``runs/figures/``):
  - aug_effect_scores.png      测试真值 + 两种模型逐窗口异常分数 (含种子)
  - aug_pr_curves.png          真实测试集上的 PR 曲线 (real-only vs 增强)
  - aug_confusion.png          混淆矩阵 (real-only vs 增强), 标注 F1/Recall/FPR
  - aug_metric_bars.png        F1/Recall/FPR/ROC-AUC/AP 对比
  - aug_ratio_sweep.png        增强比例 (生成数/真实训练数) 扫描, 0% 基线与多种子
  - aug_ratio_seed.png         最佳方法的逐种子 F1 随增强比例分布
  - aug_detector_effect.png    检测器分支: 只用真实正常训练 vs 增强后再训练
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIG = RUNS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FIXED_ROOT = RUNS / "upstream_gpu_full_20260910"
EXT_GEN_ROOT = RUNS / "all_papers_extended_20260910" / "generators"
DET_ROOT = RUNS / "augmented_upstream_detection_gpu_20260910"

FIXED_METHODS = ["CSDI", "MTSCI", "IMDiffusion", "Diffusion-TS", "D3R"]
REGIMES = ["real", "raw_1x", "raw_10x", "raw_100x", "raw_large",
           "projected_1x", "projected_10x", "projected_100x", "projected_large"]
DATES = ["250605", "250626", "250718"]
RF = "RF"

COLORS = {"Real-only": "black", "Augmented": "#d62728"}
METHOD_COLORS = {"CSDI": "#1f77b4", "MTSCI": "#2ca02c", "IMDiffusion": "#ff7f0e",
                 "Diffusion-TS": "#9467bd", "D3R": "#8c564b"}


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def result_files(root: Path):
    """Yield (method, seed, run_dir, result) for a generator root."""
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for seed_dir in sorted(method_dir.glob("seed*")):
            seed = int(seed_dir.name[4:])
            for date_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                rf = date_dir / "result.json"
                if rf.exists():
                    yield method_dir.name, seed, date_dir.name, json.loads(rf.read_text(encoding="utf-8"))


def load_metrics(root: Path):
    """DataFrame from result.json: method, seed, date, regime, n_synthetic, metrics..., plus ratio."""
    rows = []
    for method, seed, date, d in result_files(root):
        per = {r["regime"]: r for r in d.get("results", []) if r.get("classifier") == RF}
        n_fit = per.get("raw_1x", {}).get("n_synthetic")
        for regime, r in per.items():
            m = r["metrics"]
            rows.append({
                "method": method, "seed": seed, "test_date": date, "regime": regime,
                "n_synthetic": r["n_synthetic"],
                "n_real_train": n_fit,
                "ratio": 0.0 if regime == "real" or not n_fit else r["n_synthetic"] / n_fit,
                "Precision": m.get("Precision"), "Recall": m.get("Recall"), "F1": m.get("F1"),
                "FPR": np.nan,
                "ROC-AUC": m.get("ROC-AUC"), "AP": m.get("Average Precision"),
            })
    return pd.DataFrame(rows)


def load_score_metrics():
    """DataFrame recomputed from the saved real-test scores of the fixed generators.

    Contains FPR (needs the threshold and the normal-window count), which result.json does
    not store.  Only the five fixed-version generators save per-window scores.
    """
    rows = []
    for method in FIXED_METHODS:
        for seed in (42, 123, 2024):
            for date in DATES:
                per = {}
                for regime in REGIMES:
                    p = prediction_path(method, seed, date, regime)
                    if p.exists():
                        per[regime] = pd.read_csv(p)
                if "real" not in per:
                    continue
                n_fit, n_syn = None, {}
                rj = FIXED_ROOT / method / f"seed{seed}" / date / "result.json"
                if rj.exists():
                    d = json.loads(rj.read_text(encoding="utf-8"))
                    rr = {r["regime"]: r for r in d.get("results", []) if r.get("classifier") == RF}
                    n_fit = rr.get("raw_1x", {}).get("n_synthetic")
                    n_syn = {r: v.get("n_synthetic", 0) for r, v in rr.items()}
                for regime, df in per.items():
                    m = metrics_at(df.label.to_numpy(), df.score.to_numpy(), df.threshold.iloc[0])
                    ns = n_syn.get(regime, 0)
                    rows.append({
                        "method": method, "seed": seed, "test_date": date, "regime": regime,
                        "n_synthetic": ns, "n_real_train": n_fit,
                        "ratio": 0.0 if regime == "real" or not n_fit else ns / n_fit,
                        **m,
                    })
    return pd.DataFrame(rows)


def prediction_path(method, seed, date, regime, clf=RF):
    return FIXED_ROOT / method / f"seed{seed}" / date / f"{clf}_{regime}_predictions.csv"


def load_predictions(method, seed, date, regime, clf=RF):
    p = prediction_path(method, seed, date, regime, clf)
    if not p.exists():
        return None
    return pd.read_csv(p)


# --------------------------------------------------------------------------- #
#  Metric helpers
# --------------------------------------------------------------------------- #
def confusion(labels, scores, threshold):
    pred = scores >= threshold
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    return tp, fp, tn, fn


def metrics_at(labels, scores, threshold):
    tp, fp, tn, fn = confusion(labels, scores, threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"Precision": precision, "Recall": recall, "F1": f1, "FPR": fpr,
            "ROC-AUC": roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else np.nan,
            "AP": average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else np.nan}


def best_regime(metrics_df, method):
    """Pick augmented regime maximizing (ROC-AUC + AP) for RT metrics, like the report."""
    g = metrics_df[(metrics_df.method == method) & (metrics_df.regime != "real")]
    if g.empty:
        return None
    agg = g.groupby("regime")[["ROC-AUC", "AP"]].mean()
    return (agg["ROC-AUC"] + agg["AP"]).idxmax()


# --------------------------------------------------------------------------- #
#  Figure 1: test truth + anomaly scores of both models (with seeds)
# --------------------------------------------------------------------------- #
def fig_effect_scores(metrics_df, method, clf=RF, out=None):
    regime = best_regime(metrics_df, method)
    fig, axes = plt.subplots(len(DATES), 1, figsize=(15, 9.5), sharex=False)
    for ax, date in zip(axes, DATES):
        real = [load_predictions(method, s, date, "real", clf) for s in (42, 123, 2024)]
        aug = [load_predictions(method, s, date, regime, clf) for s in (42, 123, 2024)]
        real = [d for d in real if d is not None]
        aug = [d for d in aug if d is not None]
        if not real or not aug:
            continue
        base = real[0]
        x = np.arange(len(base))
        # ground-truth strip at the bottom
        ax.bar(x, np.full_like(x, 0.07, dtype=float), width=1.0, bottom=-0.13,
               color=np.where(base.label.to_numpy() == 1, "#ff9d9d", "#cfe8ff"))
        # per-seed score curves
        for d in real:
            ax.plot(x, d.score, color="black", lw=0.8, alpha=0.35)
        for d in aug:
            ax.plot(x, d.score, color=COLORS["Augmented"], lw=0.8, alpha=0.35)
        rm = np.mean([d.score.to_numpy() for d in real], axis=0)
        am = np.mean([d.score.to_numpy() for d in aug], axis=0)
        ax.plot(x, rm, color="black", lw=2.4, label="Real-only (mean of 3 seeds)")
        ax.plot(x, am, color=COLORS["Augmented"], lw=2.4, label=f"{method}+{regime} (mean of 3 seeds)")
        rt = float(np.mean([d.threshold.iloc[0] for d in real]))
        at = float(np.mean([d.threshold.iloc[0] for d in aug]))
        ax.axhline(rt, color="black", ls="--", lw=1.2, label=f"real threshold={rt:.2f}")
        ax.axhline(at, color=COLORS["Augmented"], ls="--", lw=1.2, label=f"aug threshold={at:.2f}")
        n_pos = int(base.label.sum())
        ax.set_ylabel("anomaly score")
        ax.set_title(f"held-out test date {date}  (n_test={len(base)}, n_anomaly={n_pos})")
        ax.set_ylim(-0.12, 1.05)
        ax.set_xlim(-0.5, len(base) - 0.5)
    handles = [
        Patch(facecolor="#ff9d9d", edgecolor="gray", label="ground truth = abnormal (label=1)"),
        Patch(facecolor="#cfe8ff", edgecolor="gray", label="ground truth = normal (label=0)"),
        Line2D([0], [0], color="black", lw=2.4, label="Real-only, mean of 3 seeds"),
        Line2D([0], [0], color="black", lw=0.8, alpha=0.35, label="Real-only, single seed"),
        Line2D([0], [0], color=COLORS["Augmented"], lw=2.4, label=f"{method}+{regime}, mean of 3 seeds"),
        Line2D([0], [0], color=COLORS["Augmented"], lw=0.8, alpha=0.35, label=f"{method}+{regime}, single seed"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", label="decision threshold (real)"),
        Line2D([0], [0], color=COLORS["Augmented"], lw=1.2, ls="--", label="decision threshold (aug)"),
    ]
    axes[0].legend(handles=handles, fontsize=7.5, ncol=4, loc="lower left",
                   framealpha=0.9, columnspacing=1.0)
    axes[-1].set_xlabel("test window index\n"
                        "bottom strip = ground truth (pink = abnormal, light blue = normal)")
    fig.suptitle(f"Before/after augmentation: anomaly scores on the same fixed real test set "
                 f"({method}, {clf})", y=1.0)
    fig.tight_layout()
    p = out or FIG / "aug_effect_scores.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
#  Figure 2: PR curves (fixed real test set)
# --------------------------------------------------------------------------- #
def _mean_pr(labels, score_list):
    grid = np.linspace(0, 1, 200)
    prs = []
    aps = []
    for s in score_list:
        pr, rc, _ = precision_recall_curve(labels, s)
        # rc is decreasing; interpolate onto increasing grid
        prs.append(np.interp(grid, rc[::-1], pr[::-1]))
        aps.append(average_precision_score(labels, s))
    return grid, np.mean(prs, axis=0), np.std(prs, axis=0), float(np.mean(aps))


def fig_pr_curves(metrics_df, method, clf=RF, out=None):
    regime = best_regime(metrics_df, method)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    panels = list(DATES) + ["pooled"]
    for ax, key in zip(axes, panels):
        r_scores, a_scores, labels = [], [], []
        for s in (42, 123, 2024):
            if key == "pooled":
                rl = [load_predictions(method, s, dt, "real", clf) for dt in DATES]
                al = [load_predictions(method, s, dt, regime, clf) for dt in DATES]
                if any(d is None for d in rl + al):
                    continue
                r = pd.concat(rl); a = pd.concat(al)
            else:
                r = load_predictions(method, s, key, "real", clf)
                a = load_predictions(method, s, key, regime, clf)
                if r is None or a is None:
                    continue
            labels = r.label.to_numpy()
            r_scores.append(r.score.to_numpy())
            a_scores.append(a.score.to_numpy())
        if not r_scores:
            continue
        g, rp, rs, rap = _mean_pr(labels, r_scores)
        _, ap, asd, aap = _mean_pr(labels, a_scores)
        prev = labels.mean()
        ax.plot(g, rp, color="black", lw=2.2, label=f"Real-only (AP={rap:.3f})")
        ax.fill_between(g, rp - rs, rp + rs, color="black", alpha=0.12)
        ax.plot(g, ap, color=COLORS["Augmented"], lw=2.2, label=f"{method}+{regime} (AP={aap:.3f})")
        ax.fill_between(g, ap - asd, ap + asd, color=COLORS["Augmented"], alpha=0.12)
        ax.axhline(prev, color="gray", ls=":", lw=1.0, label=f"chance (prev={prev:.2f})")
        ax.set_xlabel("Recall")
        ax.set_title(key if key != "pooled" else "pooled over 3 held-out dates")
        ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    axes[0].set_ylabel("Precision")
    fig.suptitle(f"PR curves on the fixed real test set (mean ± std band over 3 seeds, {clf})", y=1.02)
    fig.tight_layout()
    p = out or FIG / "aug_pr_curves.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
#  Figure 3: confusion matrices (real-only vs augmented)
# --------------------------------------------------------------------------- #
def _agg_confusion(method, regime, date, clf=RF):
    tot = np.zeros((2, 2), dtype=int)
    for s in (42, 123, 2024):
        d = load_predictions(method, s, date, regime, clf)
        if d is None:
            return None
        tp, fp, tn, fn = confusion(d.label.to_numpy(), d.score.to_numpy(), d.threshold.iloc[0])
        tot[0, 0] += tn; tot[0, 1] += fp; tot[1, 0] += fn; tot[1, 1] += tp
    return tot


def _cm_metrics(cm):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return precision, recall, f1, fpr


def fig_confusion(metrics_df, method, clf=RF, out=None):
    regime = best_regime(metrics_df, method)
    fig, axes = plt.subplots(2, len(DATES), figsize=(4.6 * len(DATES), 9.4),
                             constrained_layout=True)
    last_im = None
    for j, date in enumerate(DATES):
        for i, (name, reg) in enumerate([("Real-only", "real"), (f"{method}+{regime}", regime)]):
            cm = _agg_confusion(method, reg, date, clf)
            ax = axes[i, j]
            if cm is None:
                ax.axis("off"); continue
            last_im = ax.imshow(cm, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, str(cm[a, b]), ha="center", va="center",
                            color="white" if cm[a, b] > cm.max() / 2 else "black", fontsize=13)
            p, r, f1, fpr = _cm_metrics(cm)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "abnormal"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "abnormal"])
            ax.set_title(f"{name}\n{date}  F1={f1:.3f} R={r:.3f} FPR={fpr:.3f}", fontsize=10)
            ax.set_xlabel("predicted"); ax.set_ylabel("truth")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
        cbar.set_label("cell count (summed over 3 seeds)", fontsize=9)
    fig.suptitle(f"Confusion matrices on the fixed real test set (counts summed over 3 seeds, {clf})\n"
                 f"rows = truth (normal / abnormal), columns = predicted; "
                 f"R = Recall (abnormal), FPR = false-positive rate (normal)", y=1.0)
    p = out or FIG / "aug_confusion.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
#  Figure 4: metric bars real vs augmented
# --------------------------------------------------------------------------- #
def fig_metric_bars(metrics_df, method, clf=RF, out=None):
    regime = best_regime(metrics_df, method)
    rows = []
    for date in DATES:
        for s in (42, 123, 2024):
            for name, reg in [("Real-only", "real"), ("Augmented", regime)]:
                d = load_predictions(method, s, date, reg, clf)
                if d is None:
                    continue
                m = metrics_at(d.label.to_numpy(), d.score.to_numpy(), d.threshold.iloc[0])
                rows.append({"model": name, "date": date, "seed": s, "regime": reg, **m})
    df = pd.DataFrame(rows)
    metrics = ["F1", "Recall", "FPR", "ROC-AUC", "AP"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.8))
    x = np.arange(2)
    for ax, met in zip(axes, metrics):
        means, stds = [], []
        for name in ("Real-only", "Augmented"):
            v = df[df.model == name][met].to_numpy()
            means.append(np.nanmean(v)); stds.append(np.nanstd(v))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[COLORS["Real-only"], COLORS["Augmented"]], width=0.6)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(["Real-only", f"+{regime}"], fontsize=9)
        ax.set_title(met); ax.grid(axis="y", alpha=0.25)
        if met in ("ROC-AUC", "AP", "F1", "Recall"):
            ax.set_ylim(0, 1.05)
    handles = [
        Patch(facecolor=COLORS["Real-only"], label="Real-only (0% augmentation)"),
        Patch(facecolor=COLORS["Augmented"], label=f"Augmented ({method}+{regime})"),
        Line2D([0], [0], color="black", lw=0, marker="|", markersize=14, label="error bar = ±1 std"),
    ]
    fig.legend(handles=handles, fontsize=8.5, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    fig.suptitle(f"Downstream metrics on fixed real test set ({method}, {clf}): "
                 f"bars = mean, error bars = ±1 std over 3 seeds × 3 held-out dates\n"
                 f"(higher is better for F1/Recall/ROC-AUC/AP; lower is better for FPR)", y=1.10)
    fig.tight_layout()
    p = out or FIG / "aug_metric_bars.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
#  Figure 5: augmentation-ratio sweep (0% baseline + multiple seeds)
# --------------------------------------------------------------------------- #
def fig_ratio_sweep(metrics_df, out=None):
    metrics = ["F1", "Recall", "FPR", "ROC-AUC", "AP", "Precision"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    all_methods = sorted(metrics_df.method.unique())
    # faint background for non-fixed (mechanism-port) methods
    for ax, met in zip(axes.ravel(), metrics):
        for method in all_methods:
            g = metrics_df[metrics_df.method == method]
            is_fixed = method in FIXED_METHODS
            color = METHOD_COLORS.get(method, "gray")
            for seed, gs in g.groupby("seed"):
                agg = gs.groupby("ratio")[met].mean().sort_index()
                ax.scatter(agg.index + (np.random.rand(len(agg)) - 0.5) * 0.02,
                           agg.values, s=8, alpha=0.5 if is_fixed else 0.15,
                           color=color if is_fixed else "gray")
            agg = g.groupby("ratio")[met].agg(["mean", "std"]).sort_index()
            if is_fixed:
                ax.plot(agg.index, agg["mean"], "-o", color=color, lw=2, ms=4, label=method)
                ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                                color=color, alpha=0.12)
            else:
                ax.plot(agg.index, agg["mean"], "-", color="gray", lw=0.8, alpha=0.35)
        # 0% baseline marker
        base = metrics_df[metrics_df.regime == "real"][met].mean()
        ax.axhline(base, color="black", ls="--", lw=1.2, label="0% baseline (real-only)")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("synthetic / real training windows")
        ax.set_ylabel(met); ax.set_title(met); ax.grid(alpha=0.25)
        if met in ("F1", "Recall", "ROC-AUC", "AP", "Precision"):
            ax.set_ylim(0, 1.05)
    handles = []
    for m in FIXED_METHODS:
        if m in all_methods:
            handles.append(Line2D([0], [0], color=METHOD_COLORS[m], marker="o", lw=2,
                                  ms=4, label=f"{m} (fixed-version generator)"))
    handles += [
        Line2D([0], [0], color="black", lw=1.2, ls="--", label="0% baseline = real-only"),
        Line2D([0], [0], color="gray", lw=0.8, alpha=0.4,
               label="mechanism-port generator (no per-window scores → no FPR)"),
        Line2D([0], [0], color="gray", lw=0, marker="o", ms=5, alpha=0.6,
               label="one point = one seed"),
        Patch(facecolor="#c9d9f0", alpha=0.4, label="shaded band = ±1 std over seeds"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.suptitle("Augmentation-ratio sweep on fixed real test set\n"
                 "x = #synthetic ÷ #real training windows (symlog, 0% baseline at left); "
                 "line = mean over 3 seeds × 3 dates; dots = individual seeds; "
                 "gray = mechanism-port generators (result.json only)", y=1.0)
    fig.tight_layout()
    p = out or FIG / "aug_ratio_sweep.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_ratio_seed(metrics_df, method, out=None):
    g = metrics_df[(metrics_df.method == method)]
    order = sorted(g.ratio.unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, met in zip(axes, ["F1", "ROC-AUC", "AP"]):
        data = [g[g.ratio == r][met].dropna().to_numpy() for r in order]
        pos = np.arange(len(order))
        bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor("#c9d9f0")
        for i, r in enumerate(order):
            vals = data[i]
            ax.scatter(np.full(len(vals), i) + (np.random.rand(len(vals)) - 0.5) * 0.18,
                       vals, s=18, color=COLORS["Augmented"], alpha=0.7, zorder=3)
        labels = ["0%" if r == 0 else f"{r:.0f}x" if r >= 2 else f"{r:.2g}x" for r in order]
        ax.set_xticks(pos); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("synthetic / real"); ax.set_ylabel(met); ax.set_title(f"{method}: {met}")
        if met in ("F1", "ROC-AUC", "AP"):
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        if ax is axes[0]:
            handles = [
                Patch(facecolor="#c9d9f0", edgecolor="black", label="box = IQR over 3 seeds × 3 dates"),
                Line2D([0], [0], color="black", lw=0, marker="|", ms=14, label="whisker / median line"),
                Line2D([0], [0], color=COLORS["Augmented"], lw=0, marker="o", ms=6, alpha=0.7,
                       label="red dot = one run (seed × date)"),
            ]
            ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.suptitle(f"Seed variability across augmentation ratios ({method}, 3 seeds × 3 held-out dates)\n"
                 f"x-axis: 0% = real-only baseline, then the synthetic/real ratio; "
                 f"boxes show spread, not just the mean", y=1.04)
    fig.tight_layout()
    p = out or FIG / "aug_ratio_seed.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
#  Figure 6: detector branch (train on real normals vs augmented normals)
# --------------------------------------------------------------------------- #
def load_detector(detector, aug, volume, seed, date):
    p = DET_ROOT / detector / aug / volume / f"seed{seed}" / date / "test_predictions.csv"
    if not p.exists():
        return None, None
    d = pd.read_csv(p)
    return d, DET_ROOT / detector / aug / volume / f"seed{seed}" / date / "result.json"


def fig_detector_effect(out=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    # choose best augmentation (by mean AP over seeds/dates) per detector
    for row, detector in enumerate(["D3R", "IMDiffusion"]):
        combos = {}
        for aug_dir in sorted(p for p in (DET_ROOT / detector).iterdir() if p.is_dir()):
            for vol_dir in sorted(p for p in aug_dir.iterdir() if p.is_dir()):
                aps = []
                for s in (42, 123, 2024):
                    for dt in DATES:
                        d, _ = load_detector(detector, aug_dir.name, vol_dir.name, s, dt)
                        if d is not None:
                            aps.append(average_precision_score(d.label, d.score))
                if aps:
                    combos[(aug_dir.name, vol_dir.name)] = np.mean(aps)
        best = max(combos, key=combos.get)
        # PR curves: real baseline vs best augmentation, pooled per seed
        ax = axes[row, 0]
        for name, (aug, vol), color in [(f"Real-only (real normals, 1x)", ("real", "1x"), "black"),
                                        (f"+{best[0]} {best[1]} (augmented normals)", best, COLORS["Augmented"])]:
            grid = np.linspace(0, 1, 200); prs = []
            for s in (42, 123, 2024):
                rl, al = [], []
                for dt in DATES:
                    d, _ = load_detector(detector, aug, vol, s, dt)
                    if d is not None:
                        rl.append(d)
                if not rl:
                    continue
                cat = pd.concat(rl)
                pr, rc, _ = precision_recall_curve(cat.label, cat.score)
                prs.append(np.interp(grid, rc[::-1], pr[::-1]))
            if prs:
                ax.plot(grid, np.mean(prs, axis=0), color=color, lw=2.2, label=name)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"{detector} detector PR (pooled held-out dates)")
        ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower left"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
        # metric bars
        ax = axes[row, 1]
        rows = []
        for name, (aug, vol) in [("Real-only", ("real", "1x")), ("Augmented", best)]:
            for s in (42, 123, 2024):
                for dt in DATES:
                    d, _ = load_detector(detector, aug, vol, s, dt)
                    if d is None:
                        continue
                    m = metrics_at(d.label.to_numpy(), d.score.to_numpy(), d.threshold.iloc[0])
                    rows.append({"model": name, **m})
        df = pd.DataFrame(rows)
        metrics = ["F1", "Recall", "FPR", "ROC-AUC", "AP"]
        xs = np.arange(len(metrics)); w = 0.38
        for k, name in enumerate(["Real-only", "Augmented"]):
            vals = [df[df.model == name][m].mean() for m in metrics]
            errs = [df[df.model == name][m].std() for m in metrics]
            ax.bar(xs + (k - 0.5) * w, vals, w, yerr=errs, capsize=3,
                   color=[COLORS["Real-only"], COLORS["Augmented"]][k], label=name)
        ax.set_xticks(xs); ax.set_xticklabels(metrics, fontsize=8)
        ax.set_title(f"{detector}: real vs +{best[0]} {best[1]}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Detector branch: anomaly detectors trained on real normals only vs augmented normals\n"
                 "left = PR curve (pooled held-out dates); right = metrics (mean, error bar = ±1 std "
                 "over 3 seeds × 3 dates); lower is better for FPR", y=1.0)
    fig.tight_layout()
    p = out or FIG / "aug_detector_effect.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="auto",
                    help="fixed generator for the score/PR/confusion panels; auto=best AP")
    ap.add_argument("--classifier", default=RF)
    ap.add_argument("--skip-detector", action="store_true")
    args = ap.parse_args()

    metrics_df = load_metrics(FIXED_ROOT)
    if metrics_df.empty:
        raise SystemExit(f"no metrics under {FIXED_ROOT}")
    np.random.seed(0)

    # sweep table: score-recomputed fixed-method metrics (has FPR) + result.json rows for
    # the mechanism-port generators (no per-window scores, so no FPR)
    sweep_df = load_score_metrics()
    if EXT_GEN_ROOT.exists():
        ext = load_metrics(EXT_GEN_ROOT)
        ext = ext[ext.method.isin(sweep_df.method.unique()) == False] if not sweep_df.empty else ext
        sweep_df = pd.concat([sweep_df, ext], ignore_index=True)
    sweep_df.to_csv(RUNS / "aug_ratio_sweep_metrics.csv", index=False)

    method = args.method
    if method == "auto":
        agg = (metrics_df[metrics_df.regime != "real"]
               .groupby(["method", "regime"])[["ROC-AUC", "AP"]].mean())
        method, regime = (agg["ROC-AUC"] + agg["AP"]).idxmax()
        print(f"auto-selected: {method} / {regime}")

    fig_effect_scores(metrics_df, method, args.classifier)
    fig_pr_curves(metrics_df, method, args.classifier)
    fig_confusion(metrics_df, method, args.classifier)
    fig_metric_bars(metrics_df, method, args.classifier)
    fig_ratio_sweep(sweep_df)
    fig_ratio_seed(sweep_df, method)
    if not args.skip_detector and DET_ROOT.exists():
        fig_detector_effect()
    print("all augmentation-effect figures in", FIG)


if __name__ == "__main__":
    main()

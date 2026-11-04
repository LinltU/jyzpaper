"""Annotate device_data_110_220_final.xlsx and benchmark paper generators.

Anomaly labels are pseudo-labels derived from robust environment-adjusted residuals:
per device, fit Measure ~ Temp + Hum + elapsed_days with Huber regression, then
flag points with very large robust residuals or a residual plus local jump. This
does not replace manual engineering labels; it standardizes candidate outliers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
WORKBOOK = Path("/home/test/linliting/jyzmentor/device_data_110_220_final.xlsx")
OUT_DEFAULT = HERE / "runs" / "device_data_110_220_paper_aug_20260920"
FEATURES = ["Measure", "env_residual", "Hum", "Temp", "delta_measure"]
LENGTH = 4
RATIOS = {"2to8": 0.2, "3to7": 0.3, "4to6": 0.4, "5to5": 0.5}  # train:test
UPSTREAM_METHODS = ["CSDI", "MTSCI", "IMDiffusion", "Diffusion-TS", "D3R"]
MECHANISM_METHODS = ["TimeGrad", "NsDiff", "FIDE", "PaD-TS", "WaveStitch", "TabDDPM", "CCDM", "DiffPLF", "TSDE", "SSSD", "CoSTI", "SSD-TS", "MCD-TSF"]
METHODS = UPSTREAM_METHODS + MECHANISM_METHODS

DEVICE_LOCATIONS = {
    "863644079749421": "左边下相小号侧串1_220kV_18片_第三~四片",
    "863644079730603": "左边下相小号侧串1_220kV_18片_第九~十片",
    "863644079730652": "左边下相小号侧串1_220kV_18片_第十五~十六片",
    "863644079749355": "左边下相小号侧串2_220kV_18片_第三~四片",
    "863644079749231": "左边下相小号侧串2_220kV_18片_第九~十片",
    "863644079730728": "左边下相小号侧串2_220kV_18片_第十五~十六片",
    "863644079749413": "左边下相大号侧串1_220kV_18片_第三~四片",
    "863644079730611": "左边下相大号侧串1_220kV_18片_第九~十片",
    "863644079730645": "左边下相大号侧串1_220kV_18片_第十五~十六片",
    "863644079730637": "左边下相大号侧串2_220kV_18片_第三~四片",
    "863644079730694": "左边下相大号侧串2_220kV_18片_第九~十片",
    "863644079730660": "左边下相大号侧串2_220kV_18片_第十五~十六片",
    "863644079730629": "右边上相小号侧串1_220kV_18片_第三~四片",
    "863644079730744": "右边上相小号侧串1_220kV_18片_第九~十片",
    "863644079730736": "右边上相小号侧串1_220kV_18片_第十五~十六片",
    "863644079749454": "右边上相小号侧串2_220kV_18片_第三~四片",
    "863644079749249": "右边上相小号侧串2_220kV_18片_第九~十片",
    "863644079749405": "右边上相小号侧串2_220kV_18片_第十五~十六片",
    "863644079730686": "右边上相大号侧串1_220kV_18片_第三~四片",
    "863644079749397": "右边上相大号侧串1_220kV_18片_第九~十片",
    "863644079729449": "右边上相大号侧串1_220kV_18片_第十五~十六片",
    "863644079749439": "右边上相大号侧串2_220kV_18片_第九~十片",
    "863644079749363": "右边上相大号侧串2_220kV_18片_第十五~十六片",
    "863644079748878": "玉中线串1_110kV_9片_第三~四片",
    "863644079759867": "玉中线串1_110kV_9片_第六~七片",
}


def location_group(location):
    # Serial 1/2 is deliberately removed: corresponding positions on the two
    # parallel strings are the peer group used for anomaly comparison.
    return location.replace("串1_", "串X_").replace("串2_", "串X_")


def location_group_intra_string(location):
    # Keep the serial number and drop only the insulator position (片位) suffix,
    # so several devices mounted on the SAME physical string become peers.
    # Example: 左边下相小号侧串1_220kV_18片_第三~四片 -> 左边下相小号侧串1_220kV_18片
    return re.sub(r"_[^_]*片$", "", location)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def annotate_device(df, sheet_name):
    df = df.copy()
    df = df.dropna(subset=["Timestamp", "Measure"]).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    # Some rows contain a missing/zero IMEI value after Excel export. The sheet
    # name is the reliable device identifier in this workbook.
    df["IMEI"] = str(sheet_name)
    t = (df.Timestamp - df.Timestamp.min()).dt.total_seconds().to_numpy(float) / 86400.0
    xraw = np.c_[df.Temp.to_numpy(float), df.Hum.to_numpy(float), t]
    x = SimpleImputer(strategy="median").fit_transform(xraw)
    y = pd.Series(pd.to_numeric(df.Measure, errors="coerce")).ffill().bfill().to_numpy(float)
    sc = StandardScaler().fit(x)
    model = HuberRegressor(epsilon=1.5, alpha=1e-4, max_iter=300).fit(sc.transform(x), y)
    pred = model.predict(sc.transform(x))
    residual = y - pred
    med = np.median(residual)
    mad = np.median(np.abs(residual - med)) * 1.4826 + 1e-9
    z = (residual - med) / mad
    diff = np.r_[0.0, np.diff(y)]
    dmed = np.median(diff)
    dmad = np.median(np.abs(diff - dmed)) * 1.4826 + 1e-9
    jump_z = np.abs(diff - dmed) / dmad
    label = (np.abs(z) >= 5.0) | ((np.abs(z) >= 3.5) & (jump_z >= 4.0))
    df["elapsed_days"] = t
    df["env_pred_measure"] = pred
    df["env_residual"] = residual
    df["robust_residual_z"] = z
    df["delta_measure"] = diff
    df["jump_z"] = jump_z
    df["pseudo_anomaly"] = label.astype(int)
    df["pseudo_label_rule"] = "abs(env_residual_z)>=5 OR abs(env_residual_z)>=3.5 AND jump_z>=4"
    return df


def load_annotated():
    sheets = pd.read_excel(WORKBOOK, sheet_name=None)
    frames = [annotate_device(df, name) for name, df in sheets.items()]
    data = pd.concat(frames, ignore_index=True)
    data["Location"] = data["IMEI"].map(DEVICE_LOCATIONS)
    data["group_id"] = data["Location"].map(location_group)
    data = apply_peer_labels(data)
    summary = data.groupby("IMEI").agg(
        rows=("IMEI", "size"),
        anomaly_points=("pseudo_anomaly", "sum"),
        anomaly_rate=("pseudo_anomaly", "mean"),
        start=("Timestamp", "min"),
        end=("Timestamp", "max"),
        measure_min=("Measure", "min"),
        measure_max=("Measure", "max"),
        residual_z_max=("robust_residual_z", lambda s: float(np.max(np.abs(s)))),
    ).reset_index()
    return data, summary


def apply_peer_labels(data):
    """Add same-string corresponding-position residual labels.

    For each corresponding-position group, a device is compared with the
    nearest contemporaneous peer from the other serial. The peer residual is
    computed after each-device temperature/humidity/time calibration, so the
    rule targets deviations from the shared string behavior rather than raw
    level differences. Groups without a peer retain the individual robust rule.
    """
    data = data.copy()
    data["peer_residual"] = np.nan
    data["peer_delta"] = np.nan
    data["peer_z"] = np.nan
    data["peer_match_seconds"] = np.nan
    for group_id, group in data.groupby("group_id", sort=False):
        devices = list(group.IMEI.unique())
        if len(devices) < 2:
            continue
        for imei in devices:
            target = group[group.IMEI == imei].sort_values("Timestamp")
            peers = group[group.IMEI != imei].sort_values("Timestamp")
            if peers.empty:
                continue
            j = np.searchsorted(peers.Timestamp.to_numpy(dtype="datetime64[ns]"), target.Timestamp.to_numpy(dtype="datetime64[ns]"))
            left = np.clip(j - 1, 0, len(peers) - 1)
            right = np.clip(j, 0, len(peers) - 1)
            t_ns = target.Timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
            p_ns = peers.Timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
            use_right = np.abs(p_ns[right] - t_ns) < np.abs(p_ns[left] - t_ns)
            chosen = np.where(use_right, right, left)
            seconds = np.abs(p_ns[chosen] - t_ns) / 1e9
            peer_res = peers.env_residual.to_numpy(float)[chosen]
            delta = target.env_residual.to_numpy(float) - peer_res
            valid = seconds <= 1800
            idx = target.index.to_numpy()
            data.loc[idx[valid], "peer_residual"] = peer_res[valid]
            data.loc[idx[valid], "peer_delta"] = delta[valid]
            data.loc[idx[valid], "peer_match_seconds"] = seconds[valid]
    for group_id, group in data.groupby("group_id", sort=False):
        valid = group.peer_delta.notna()
        if valid.sum() < 8:
            continue
        vals = group.loc[valid, "peer_delta"].to_numpy(float)
        med = np.median(vals)
        mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-9
        data.loc[group.index[valid], "peer_z"] = (vals - med) / mad
    individual = data.robust_residual_z.abs() >= 5.0
    peer = data.peer_z.abs() >= 5.0
    combined = individual | peer
    data["pseudo_anomaly_individual"] = individual.astype(int)
    data["pseudo_anomaly_peer"] = peer.astype(int)
    data["pseudo_anomaly"] = combined.astype(int)
    data["pseudo_label_rule"] = "individual_abs_z>=5 OR same_string_peer_abs_z>=5; peer=nearest corresponding position within 30min"
    return data


def build_windows(data):
    rows = []
    meta = []
    for imei, g in data.groupby("IMEI", sort=False):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        arr = g[FEATURES].to_numpy(np.float32)
        lab = g.pseudo_anomaly.to_numpy(int)
        for start in range(max(0, len(g) - LENGTH + 1)):
            z = arr[start:start + LENGTH]
            labs = lab[start:start + LENGTH]
            rows.append(z)
            meta.append({
                "IMEI": imei,
                "Location": g.Location.iloc[0],
                "group_id": g.group_id.iloc[0],
                "start_time": str(g.Timestamp.iloc[start]),
                "end_time": str(g.Timestamp.iloc[start + LENGTH - 1]),
                "label": int(labs.max()),
                "point_anomaly_count": int(labs.sum()),
                "source_rows": list(range(start, start + LENGTH)),
            })
    return np.asarray(rows, np.float32), pd.DataFrame(meta)


def split_ratio(meta, train_frac):
    fit, val, test = [], [], []
    for _, g in meta.groupby("IMEI", sort=False):
        ids = g.index.to_numpy()
        cut = max(1, min(len(ids) - 1, int(len(ids) * train_frac))) if len(ids) > 1 else 1
        train_ids, test_ids = ids[:cut], ids[cut:]
        if len(train_ids) > 4:
            vcut = max(1, int(len(train_ids) * 0.8))
            fit.extend(train_ids[:vcut])
            val.extend(train_ids[vcut:])
        else:
            fit.extend(train_ids)
        test.extend(test_ids)
    return np.asarray(fit, int), np.asarray(val, int), np.asarray(test, int)


def clf_features(x):
    return np.concatenate([x[:, -1], x.mean(1), x.std(1), x[:, -1] - x[:, 0]], axis=1)


def threshold(y, p):
    candidates = np.unique(np.r_[p, 0.5])
    return float(max(candidates, key=lambda t: (f1_score(y, p >= t, zero_division=0), t)))


def evaluate(xfit, yfit, xval, yval, xtest, ytest, seed):
    out = []
    for name in ("RF", "HGB"):
        if name == "RF":
            model = RandomForestClassifier(n_estimators=180, max_depth=9, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=2)
        else:
            model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=9, min_samples_leaf=5, l2_regularization=1, random_state=seed)
        model.fit(clf_features(xfit), yfit)
        pv = model.predict_proba(clf_features(xval))[:, 1] if len(xval) else model.predict_proba(clf_features(xfit))[:, 1]
        pe = model.predict_proba(clf_features(xtest))[:, 1]
        th = threshold(yval if len(xval) else yfit, pv)
        pred = pe >= th
        out.append({
            "classifier": name,
            "Precision": precision_score(ytest, pred, zero_division=0),
            "Recall": recall_score(ytest, pred, zero_division=0),
            "F1": f1_score(ytest, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(ytest, pe) if len(np.unique(ytest)) == 2 else np.nan,
            "Average Precision": average_precision_score(ytest, pe) if ytest.sum() else np.nan,
            "threshold": th,
        })
    return out


def plot_labels(data, out):
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    devices = list(data.IMEI.drop_duplicates())
    ncols = 5
    nrows = int(np.ceil(len(devices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 3.1 * nrows), sharex=False, sharey=False)
    axes = np.asarray(axes).ravel()
    for ax, imei in zip(axes, devices):
        g = data[data.IMEI == imei]
        bad = g.pseudo_anomaly.to_numpy(bool)
        ax.plot(g.Timestamp, g.Measure, color="#64748b", lw=0.7, alpha=0.65)
        ax.scatter(g.Timestamp[~bad], g.Measure[~bad], color="#2563eb", s=5, alpha=0.55, label="normal")
        ax.scatter(g.Timestamp[bad], g.Measure[bad], color="#dc2626", s=12, alpha=0.9, label="pseudo anomaly")
        ax.set_title(f"{imei}\nanom={int(bad.sum())}/{len(g)}", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", labelrotation=30, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(devices):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Pseudo anomaly labels from robust environment-adjusted residuals", fontsize=15, y=1.035)
    fig.tight_layout()
    fig.savefig(fig_dir / "pseudo_anomaly_labels_by_device.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_results(out):
    fig_dir = out / "figures"
    result = pd.read_csv(out / "all_results.csv")
    aug = result[result.method != "Real-only"].copy()
    aug["score"] = aug["ROC-AUC"] + aug["Average Precision"]
    best = aug.loc[aug.groupby("ratio")["score"].idxmax()].set_index("ratio").loc[list(RATIOS)].reset_index()
    base = result[result.method == "Real-only"].drop_duplicates(["ratio", "classifier"])
    metrics = ["F1", "ROC-AUC", "Average Precision"]
    labels = {"F1": "F1", "ROC-AUC": "ROC-AUC", "Average Precision": "AP"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
    for ax, (_, row) in zip(axes, best.iterrows()):
        b = base[(base.ratio == row.ratio) & (base.classifier == row.classifier)].iloc[0]
        x = np.arange(len(metrics)); width = 0.36
        real = [b[m] for m in metrics]; vals = [row[m] for m in metrics]
        ax.bar(x - width/2, real, width, color="#7b8794", label="Real-only")
        ax.bar(x + width/2, vals, width, color="#2563eb", label="Augmented")
        ax.set_xticks(x, [labels[m] for m in metrics])
        ax.set_title(f"{row.ratio} train:test\n{row.method}+{row.classifier}", fontsize=10)
        ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score"); axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Device-data anomaly benchmark: Real-only vs best paper augmentation", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "real_only_vs_best_paper_augmentation.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    pivot = aug.pivot_table(index="method", columns="ratio", values="Average Precision", aggfunc="mean").loc[METHODS, list(RATIOS)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Average Precision by method and train:test ratio")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "method_ap_heatmap.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seeds[0])
    data, point_summary = load_annotated()
    data.to_csv(out / "annotated_device_points.csv", index=False, encoding="utf-8-sig")
    point_summary.to_csv(out / "annotation_summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "annotated_device_data.xlsx", engine="openpyxl") as writer:
        point_summary.to_excel(writer, sheet_name="summary", index=False)
        data.to_excel(writer, sheet_name="annotated_points", index=False)
    plot_labels(data, out)

    x, meta = build_windows(data)
    meta.to_json(out / "window_metadata.json", orient="records", force_ascii=False, indent=2)
    np.savez_compressed(out / "window_dataset.npz", x=x, y=meta.label.to_numpy(int))
    (out / "protocol.json").write_text(json.dumps({
        "source": str(WORKBOOK), "label_rule": "individual Huber environment residual robust z OR same-string corresponding-position peer residual robust z within 30 minutes", "ratios_definition": "train:test",
        "ratios": RATIOS, "features": FEATURES, "window_length": LENGTH, "methods": METHODS,
        "epochs": args.epochs, "steps": args.steps, "seeds": args.seeds, "multiplier": args.multiplier,
        "device": args.device, "generation_scope": args.scope, "min_per_scope": args.min_per_scope,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    mech = load_mod("device_mech", HERE / "all_papers_extended.py")
    upstream = load_mod("device_upstream", HERE / "upstream_large_augmentation.py")
    all_rows, train_rows = [], []
    for ratio_name, train_frac in RATIOS.items():
        fit, val, test = split_ratio(meta, train_frac)
        yfit = meta.iloc[fit].label.to_numpy(int); yval = meta.iloc[val].label.to_numpy(int); ytest = meta.iloc[test].label.to_numpy(int)
        if len(np.unique(ytest)) < 2 or len(np.unique(yfit)) < 2:
            continue
        for method in METHODS:
            for seed in args.seeds:
                seed_all(seed)
                scaler = MinMaxScaler(feature_range=(-0.8, 0.8)).fit(x[fit].reshape(-1, len(FEATURES)))
                z = scaler.transform(x.reshape(-1, len(FEATURES))).reshape(x.shape).astype(np.float32)
                generated, gen_labels, gen_sources = [], [], []
                gen_meta = []
                for label in (0, 1):
                    label_fit = fit[yfit == label]
                    if not len(label_fit):
                        continue
                    if args.scope == "pooled":
                        scope_items = [("pooled", label_fit)]
                    elif args.scope == "peer_group":
                        grouped = meta.iloc[label_fit].groupby("group_id").groups
                        scope_items = [(str(key), label_fit[np.isin(label_fit, ids.to_numpy())]) for key, ids in grouped.items()]
                    else:
                        grouped = meta.iloc[label_fit].groupby("IMEI").groups
                        scope_items = [(str(key), label_fit[np.isin(label_fit, ids.to_numpy())]) for key, ids in grouped.items()]
                    for scope_key, ids in scope_items:
                        if not len(ids):
                            continue
                        scope_hash = int(hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:8], 16)
                        local_seed = seed + label + scope_hash % 100000
                        if method in UPSTREAM_METHODS:
                            model = upstream.Upstream(method, args.steps, args.hidden, args.device)
                            opt = torch.optim.Adam(model.model.parameters(), lr=0.001)
                            inp = torch.from_numpy(z[ids]); losses = []
                            for _ in range(args.epochs):
                                order = torch.randperm(len(inp)); batch_losses = []
                                for st in range(0, len(inp), 32):
                                    sel = order[st:st+32]
                                    loss = model.loss(inp[sel]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1); opt.step(); batch_losses.append(float(loss.detach()))
                                losses.append(float(np.mean(batch_losses)))
                            n = max(args.min_per_scope, len(ids) * args.multiplier)
                            chosen = np.random.default_rng(local_seed).choice(ids, n, replace=True)
                            chunks = []
                            for st in range(0, n, args.batch):
                                chunks.append(model.sample(torch.from_numpy(z[chosen[st:st+args.batch]])).detach().cpu().numpy())
                            syn = scaler.inverse_transform(np.concatenate(chunks).reshape(-1, len(FEATURES))).reshape(-1, LENGTH, len(FEATURES))
                        else:
                            model = mech.MechanismGenerator(method, args.steps, args.hidden, local_seed)
                            model.device = torch.device(args.device)
                            model.ddpm = model.ddpm.to(model.device)
                            opt = torch.optim.Adam(model.ddpm.parameters(), lr=0.001)
                            inp = torch.from_numpy(z[ids]); losses = []
                            for _ in range(args.epochs):
                                order = torch.randperm(len(inp)); batch_losses = []
                                for st in range(0, len(inp), 32):
                                    sel = order[st:st+32]
                                    loss = model.loss(inp[sel], label); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.ddpm.parameters(), 1); opt.step(); batch_losses.append(float(loss.detach()))
                                losses.append(float(np.mean(batch_losses)))
                            n = max(args.min_per_scope, len(ids) * args.multiplier)
                            chosen = np.random.default_rng(local_seed).choice(ids, n, replace=True)
                            syn = model.sample(torch.from_numpy(z[chosen]), label, n).detach().cpu().numpy()
                            syn = scaler.inverse_transform(syn.reshape(-1, len(FEATURES))).reshape(-1, LENGTH, len(FEATURES))
                        generated.append(syn.astype(np.float32)); gen_labels.extend([label] * len(syn)); gen_sources.extend(chosen.tolist())
                        gen_meta.extend([{
                            "ratio": ratio_name, "method": method, "seed": seed, "label": label,
                            "scope": args.scope, "scope_key": scope_key, "source_id": int(source_id),
                            "IMEI": meta.iloc[source_id].IMEI, "Location": meta.iloc[source_id].Location,
                            "group_id": meta.iloc[source_id].group_id, "start_time": meta.iloc[source_id].start_time,
                            "end_time": meta.iloc[source_id].end_time,
                        } for source_id in chosen])
                        train_rows.append({"ratio": ratio_name, "method": method, "seed": seed, "label": label, "scope": args.scope, "scope_key": scope_key, "n_train": len(ids), "n_generated": len(syn), "loss": losses[-1]})
                synthetic = np.concatenate(generated); sy = np.asarray(gen_labels, int)
                safe_method = method.replace("/", "_").replace("-", "_")
                generated_frame = pd.DataFrame(gen_meta)
                np.savez_compressed(
                    out / f"raw_{ratio_name}_{safe_method}_seed{seed}.npz",
                    windows=synthetic,
                    labels=sy,
                    source_ids=np.asarray(gen_sources, int),
                    source_imei=generated_frame["IMEI"].to_numpy(object),
                    source_location=generated_frame["Location"].to_numpy(object),
                    source_group_id=generated_frame["group_id"].to_numpy(object),
                    source_start_time=generated_frame["start_time"].to_numpy(object),
                    source_end_time=generated_frame["end_time"].to_numpy(object),
                )
                generated_frame.to_csv(out / f"generated_metadata_{ratio_name}_{safe_method}_seed{seed}.csv", index=False, encoding="utf-8-sig")
                for row in evaluate(x[fit], yfit, x[val], yval, x[test], ytest, seed):
                    row.update({"ratio": ratio_name, "method": "Real-only", "mode": "real", "seed": seed, "n_synthetic": 0})
                    all_rows.append(row)
                xx = np.concatenate([x[fit], synthetic]); yy = np.r_[yfit, sy]
                for row in evaluate(xx, yy, x[val], yval, x[test], ytest, seed):
                    row.update({"ratio": ratio_name, "method": method, "mode": "raw", "seed": seed, "n_synthetic": len(synthetic)})
                    all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(out / "all_results.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(train_rows).to_csv(out / "training_audit.csv", index=False, encoding="utf-8-sig")
    results = pd.DataFrame(all_rows)
    summary = results.groupby(["ratio", "method", "mode", "classifier"]).agg(
        runs=("F1", "size"), Precision=("Precision", "mean"), Recall=("Recall", "mean"), F1=("F1", "mean"),
        **{"ROC-AUC": ("ROC-AUC", "mean"), "Average Precision": ("Average Precision", "mean")}
    ).reset_index().sort_values(["ratio", "Average Precision"], ascending=[True, False])
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    plot_results(out)
    print("wrote", out)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seeds", default="42", help="comma-separated seeds, e.g. 42,123,2024")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scope", choices=("pooled", "peer_group", "device"), default="peer_group")
    parser.add_argument("--min-per-scope", type=int, default=30)
    parsed = parser.parse_args()
    parsed.seeds = [int(x.strip()) for x in parsed.seeds.split(",") if x.strip()]
    main(parsed)

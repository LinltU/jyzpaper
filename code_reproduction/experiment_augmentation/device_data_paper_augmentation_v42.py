"""Paper-code SOTA augmentation benchmark driven by v4.2 anomaly labels.

This reuses the full pipeline of ``device_data_paper_augmentation.py`` (five
window features, length-4 windows, the 18 paper generators, train:test ratios
2:8/3:7/4:6/5:5, RF/HGB downstream classifiers) but replaces the point-level
pseudo labels with the v4 time/humidity-conditioned window labels produced by
``process_device_nonoverlap_windows_v42.py``.

Label mapping (as agreed):
  * a point is anomalous when it falls inside a v4 candidate (32-point) window;
  * a length-4 sliding window is anomalous when any of its points is anomalous.

Only the generator/label construction is new; the split, feature aggregation,
generation loop and metrics are imported from ``device_data_paper_augmentation``
so the two benchmarks are directly comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import device_data_paper_augmentation as paper
import process_device_nonoverlap_windows_v42 as v4

HERE = Path(__file__).resolve().parent

# Two compatible 5-channel feature sets (all upstream model constructors are
# fixed at 5 channels, so the v4-evidence set keeps exactly five features):
#   base   : the original device_data_paper_augmentation feature set;
#   v4-evi : swaps the raw env context for the v4 point-level evidence so the
#            classifier can actually see the peer deviation that defines most
#            v4 candidates (env residual z, peer delta z, peer jump z).
FEATURE_SETS = {
    "base": ["Measure", "env_residual", "Hum", "Temp", "delta_measure"],
    "v4-evi": ["Measure", "env_residual_z", "peer_delta_z", "peer_delta_jump_z", "delta_measure"],
}
# peer_delta_jump_z is sparse and heavy-tailed (a few isolated jumps reach
# z~15+). Clip it before the shared MinMax scaler so one extreme value cannot
# squash the whole channel; the single-jump rule threshold is 12, so 20 keeps
# the discriminating region intact.
JUMP_Z_CLIP = 20.0


def v4_labelled_points(workbook: Path, peer_group_mode: str, window_size: int, baseline: str = "archived"):
    """Run the v4 pipeline and expand its window labels to point-level labels."""
    points, _models = v4.add_point_features(
        v4.load_points(workbook, peer_group_mode), 16, baseline)
    windows = v4.make_windows(points, window_size)
    label_col = f"window_anomaly_{v4.VERSION}"

    point_label = pd.Series(0, index=points.index, dtype=int)
    for imei, g in points.groupby("IMEI", sort=False):
        g = g.sort_values("Timestamp")
        gw = windows[windows.IMEI == imei].sort_values("window_number")
        arr = np.zeros(len(g), dtype=int)
        for _, row in gw.iterrows():
            if int(row[label_col]) == 1:
                arr[int(row.start_row):int(row.end_row) + 1] = 1
        point_label.loc[g.index] = arr

    points = points.copy()
    points["v4_point_label"] = point_label.to_numpy(int)
    return points, windows


def build_data(points: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Assemble the columns expected by ``paper.build_windows``.

    Features selected by ``feature_set`` must all be finite before windowing;
    peer evidence is undefined where no peer was matched, so it is filled with a
    neutral 0 (standard-score space) and the sparse jump channel is clipped.
    """
    data = points.sort_values(["IMEI", "Timestamp"]).copy()
    data["delta_measure"] = data.groupby("IMEI")["Measure"].diff().fillna(0.0)
    data["pseudo_anomaly"] = data["v4_point_label"].astype(int)
    for col in FEATURE_SETS[feature_set]:
        if col not in data:
            raise KeyError(f"feature {col!r} missing from v4 point features")
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if "peer_delta_jump_z" in FEATURE_SETS[feature_set]:
        data["peer_delta_jump_z"] = data["peer_delta_jump_z"].clip(upper=JUMP_Z_CLIP)
    data[FEATURE_SETS[feature_set]] = data[FEATURE_SETS[feature_set]].fillna(0.0)
    return data


def main(args) -> None:
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    paper.seed_all(args.seeds[0])

    points, windows = v4_labelled_points(args.workbook, args.peer_group_mode, args.window_size, args.baseline)
    points.to_csv(out / "annotated_device_points_v4.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(out / "v4_window_labels.csv", index=False, encoding="utf-8-sig")

    data = build_data(points, args.feature_set)
    paper.FEATURES = list(FEATURE_SETS[args.feature_set])
    x, meta = paper.build_windows(data)
    meta.to_json(out / "window_metadata_v4.json", orient="records", force_ascii=False, indent=2)
    np.savez_compressed(out / "window_dataset_v4.npz", x=x, y=meta.label.to_numpy(int))

    ratios = {name: paper.RATIOS[name] for name in args.ratios}
    methods = args.methods
    (out / "protocol.json").write_text(json.dumps({
        "source": str(args.workbook),
        "label_source": f"process_device_nonoverlap_windows_{v4.VERSION}.py"
        ,
        "label_rule": "point inside a v4 candidate window is anomalous; a length-4 window is anomalous if any point is anomalous",
        "v4_rule": f"see protocol_{v4.VERSION}.json; windows {args.window_size} points, peer mode {args.peer_group_mode}",
        "ratios_definition": "train:test",
        "feature_set": args.feature_set, "baseline": args.baseline,
        "ratios": ratios, "features": paper.FEATURES, "window_length": paper.LENGTH,
        "methods": methods, "epochs": args.epochs, "steps": args.steps, "seeds": args.seeds,
        "multiplier": args.multiplier, "device": args.device, "generation_scope": args.scope,
        "min_per_scope": args.min_per_scope,
        "point_anomaly_rate": float(data.pseudo_anomaly.mean()),
        "window_anomaly_rate": float(meta.label.mean()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    mech = paper.load_mod("device_mech", HERE / "all_papers_extended.py")
    upstream = paper.load_mod("device_upstream", HERE / "upstream_large_augmentation.py")
    results_path = out / "all_results.csv"
    audit_path = out / "training_audit.csv"
    all_rows, train_rows = [], []

    for ratio_name, train_frac in ratios.items():
        fit, val, test = paper.split_ratio(meta, train_frac)
        yfit = meta.iloc[fit].label.to_numpy(int)
        yval = meta.iloc[val].label.to_numpy(int)
        ytest = meta.iloc[test].label.to_numpy(int)
        if len(np.unique(ytest)) < 2 or len(np.unique(yfit)) < 2:
            print(f"skip {ratio_name}: single-class test/fit split")
            continue
        for method in methods:
            for seed in args.seeds:
                paper.seed_all(seed)
                scaler = paper.MinMaxScaler(feature_range=(-0.8, 0.8)).fit(x[fit].reshape(-1, len(paper.FEATURES)))
                z = scaler.transform(x.reshape(-1, len(paper.FEATURES))).reshape(x.shape).astype(np.float32)
                generated, gen_labels, gen_sources, gen_meta = [], [], [], []
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
                        if method in paper.UPSTREAM_METHODS:
                            model = upstream.Upstream(method, args.steps, args.hidden, args.device)
                            opt = torch.optim.Adam(model.model.parameters(), lr=0.001)
                            inp = torch.from_numpy(z[ids])
                            losses = []
                            for _ in range(args.epochs):
                                order = torch.randperm(len(inp))
                                batch_losses = []
                                for st in range(0, len(inp), 32):
                                    sel = order[st:st + 32]
                                    loss = model.loss(inp[sel])
                                    opt.zero_grad(); loss.backward()
                                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1); opt.step()
                                    batch_losses.append(float(loss.detach()))
                                losses.append(float(np.mean(batch_losses)))
                            n = max(args.min_per_scope, len(ids) * args.multiplier)
                            chosen = np.random.default_rng(local_seed).choice(ids, n, replace=True)
                            chunks = []
                            for st in range(0, n, args.batch):
                                chunks.append(model.sample(torch.from_numpy(z[chosen[st:st + args.batch]])).detach().cpu().numpy())
                            syn = scaler.inverse_transform(np.concatenate(chunks).reshape(-1, len(paper.FEATURES))).reshape(-1, paper.LENGTH, len(paper.FEATURES))
                        else:
                            model = mech.MechanismGenerator(method, args.steps, args.hidden, local_seed)
                            model.device = torch.device(args.device)
                            model.ddpm = model.ddpm.to(model.device)
                            opt = torch.optim.Adam(model.ddpm.parameters(), lr=0.001)
                            inp = torch.from_numpy(z[ids])
                            losses = []
                            for _ in range(args.epochs):
                                order = torch.randperm(len(inp))
                                batch_losses = []
                                for st in range(0, len(inp), 32):
                                    sel = order[st:st + 32]
                                    loss = model.loss(inp[sel], label)
                                    opt.zero_grad(); loss.backward()
                                    torch.nn.utils.clip_grad_norm_(model.ddpm.parameters(), 1); opt.step()
                                    batch_losses.append(float(loss.detach()))
                                losses.append(float(np.mean(batch_losses)))
                            n = max(args.min_per_scope, len(ids) * args.multiplier)
                            chosen = np.random.default_rng(local_seed).choice(ids, n, replace=True)
                            syn = model.sample(torch.from_numpy(z[chosen]), label, n).detach().cpu().numpy()
                            syn = scaler.inverse_transform(syn.reshape(-1, len(paper.FEATURES))).reshape(-1, paper.LENGTH, len(paper.FEATURES))
                        generated.append(syn.astype(np.float32))
                        gen_labels.extend([label] * len(syn))
                        gen_sources.extend(chosen.tolist())
                        gen_meta.extend([{
                            "ratio": ratio_name, "method": method, "seed": seed, "label": label,
                            "scope": args.scope, "scope_key": scope_key, "source_id": int(source_id),
                            "IMEI": meta.iloc[source_id].IMEI, "Location": meta.iloc[source_id].Location,
                            "group_id": meta.iloc[source_id].group_id, "start_time": meta.iloc[source_id].start_time,
                            "end_time": meta.iloc[source_id].end_time,
                        } for source_id in chosen])
                        train_rows.append({"ratio": ratio_name, "method": method, "seed": seed, "label": label,
                                           "scope": args.scope, "scope_key": scope_key, "n_train": len(ids),
                                           "n_generated": len(syn), "loss": losses[-1]})
                synthetic = np.concatenate(generated)
                sy = np.asarray(gen_labels, int)
                safe_method = method.replace("/", "_").replace("-", "_")
                generated_frame = pd.DataFrame(gen_meta)
                np.savez_compressed(
                    out / f"raw_{ratio_name}_{safe_method}_seed{seed}.npz",
                    windows=synthetic, labels=sy, source_ids=np.asarray(gen_sources, int),
                    source_imei=generated_frame["IMEI"].to_numpy(object),
                    source_location=generated_frame["Location"].to_numpy(object),
                    source_group_id=generated_frame["group_id"].to_numpy(object),
                    source_start_time=generated_frame["start_time"].to_numpy(object),
                    source_end_time=generated_frame["end_time"].to_numpy(object),
                )
                generated_frame.to_csv(out / f"generated_metadata_{ratio_name}_{safe_method}_seed{seed}.csv",
                                       index=False, encoding="utf-8-sig")
                for row in paper.evaluate(x[fit], yfit, x[val], yval, x[test], ytest, seed):
                    row.update({"ratio": ratio_name, "method": "Real-only", "mode": "real", "seed": seed, "n_synthetic": 0})
                    all_rows.append(row)
                xx = np.concatenate([x[fit], synthetic])
                yy = np.r_[yfit, sy]
                for row in paper.evaluate(xx, yy, x[val], yval, x[test], ytest, seed):
                    row.update({"ratio": ratio_name, "method": method, "mode": "raw", "seed": seed, "n_synthetic": len(synthetic)})
                    all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(results_path, index=False, encoding="utf-8-sig")
                pd.DataFrame(train_rows).to_csv(audit_path, index=False, encoding="utf-8-sig")

    results = pd.DataFrame(all_rows)
    summary = results.groupby(["ratio", "method", "mode", "classifier"]).agg(
        runs=("F1", "size"), Precision=("Precision", "mean"), Recall=("Recall", "mean"), F1=("F1", "mean"),
        **{"ROC-AUC": ("ROC-AUC", "mean"), "Average Precision": ("Average Precision", "mean")}
    ).reset_index().sort_values(["ratio", "Average Precision"], ascending=[True, False])
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")

    if set(ratios) == set(paper.RATIOS) and set(methods) == set(paper.METHODS):
        try:
            paper.plot_results(out)
        except Exception as exc:  # plotting is best-effort for smoke runs
            print("plot_results skipped:", exc)
    print("wrote", out)
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=paper.WORKBOOK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--peer-group-mode", choices=("intra-string", "cross-string"), default="intra-string")
    parser.add_argument("--baseline", default="archived",
                        choices=["archived", "nlinear", "huber", "last"],
                        help="v4.2 environment baseline used to derive labels")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="base",
                        help="base keeps the original 5 features; v4-evi swaps them for v4 point-level evidence")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seeds", default="42,123,2024", help="comma-separated seeds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scope", choices=("pooled", "peer_group", "device"), default="peer_group")
    parser.add_argument("--min-per-scope", type=int, default=30)
    parser.add_argument("--ratios", nargs="+", default=list(paper.RATIOS), choices=list(paper.RATIOS))
    parser.add_argument("--methods", nargs="+", default=list(paper.METHODS), choices=list(paper.METHODS))
    parsed = parser.parse_args()
    parsed.seeds = [int(s.strip()) for s in parsed.seeds.split(",") if s.strip()]
    main(parsed)

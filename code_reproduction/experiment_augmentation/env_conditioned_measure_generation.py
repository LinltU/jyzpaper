"""Environment-conditioned Measure generation test.

Fixed condition channels: Temp4, Hum4, Temp7, Hum7, elapsed_min.
Generated target channels: Measure4 residual and Measure7 residual.

This is an isolated test script; it does not modify the existing upstream run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openpyxl.styles import Alignment, Font, PatternFill
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXED = HERE / "runs" / "upstream_gpu_full_20260910"
DEFAULT_OUT = HERE / "runs" / "env_conditioned_measure_generation_20260916"
DATES = ("250605", "250626", "250718")
LENGTH = 4
FEATURES = ["target4", "target7", "Temp4", "Hum4", "Temp7", "Hum7", "elapsed_min"]
DETECTION_FEATURES = ["M4_residual", "M7_residual", "M4-M7", "M4/M7", "log(M4/M7)"]


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fit_ols(df, measure_col, temp_col, hum_col, elapsed_col):
    x = np.column_stack([
        np.ones(len(df)),
        df[temp_col].to_numpy(float),
        df[hum_col].to_numpy(float),
        df[elapsed_col].to_numpy(float),
    ])
    y = df[measure_col].to_numpy(float)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def predict_ols(beta, temp, hum, elapsed):
    x = np.column_stack([np.ones(len(temp)), temp, hum, elapsed])
    return x @ np.asarray(beta)


def make_detection(m4, m7, base4, base7):
    m4 = np.asarray(m4, dtype=float)
    m7 = np.asarray(m7, dtype=float)
    safe4 = np.maximum(m4, 1e-6)
    safe7 = np.maximum(m7, 1e-6)
    return np.stack([safe4 - base4, safe7 - base7, safe4 - safe7, safe4 / safe7, np.log(safe4 / safe7)], axis=-1)


def prepare(out, residual_mode="regression", target_mode="residual"):
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_json(FIXED / "metadata.json", dtype={"date": str}).reset_index(drop=True)
    audit = pd.read_csv(FIXED / "paired_row_audit.csv", dtype={"date": str})
    audit["dt"] = pd.to_datetime(audit["dt"])
    audit["elapsed_min"] = 0.0
    for date, ids in audit.groupby("date").groups.items():
        t0 = audit.loc[ids, "dt"].min()
        audit.loc[ids, "elapsed_min"] = (audit.loc[ids, "dt"] - t0).dt.total_seconds() / 60.0

    row_lookup = {(str(r.date), int(r.row4)): r for r in audit.itertuples(index=False)}
    regressions = {}
    for date in DATES:
        calib = audit[(audit.date == date) & (audit.role == "calibration")].copy()
        if len(calib) < 4:
            raise ValueError(f"{date}: need calibration rows for environment regression")
        base = meta[meta.date == date][["base4", "base7"]].iloc[0].to_numpy(float)
        if residual_mode == "regression":
            regressions[date] = {
                "mode": residual_mode,
                "m4_beta": fit_ols(calib, "Measure4", "Temp4", "Hum4", "elapsed_min").tolist(),
                "m7_beta": fit_ols(calib, "Measure7", "Temp7", "Hum7", "elapsed_min").tolist(),
            }
        elif residual_mode == "baseline":
            regressions[date] = {"mode": residual_mode, "base4": float(base[0]), "base7": float(base[1])}
        else:
            raise ValueError(residual_mode)

    x_env, x_det, abs_measure, cond, pred, rows = [], [], [], [], [], []
    for i, r in meta.iterrows():
        block = [row_lookup[(str(r.date), int(row4))] for row4 in r.row4]
        q = pd.DataFrame([b._asdict() for b in block])
        reg = regressions[str(r.date)]
        if reg["mode"] == "regression":
            p4 = predict_ols(reg["m4_beta"], q.Temp4.to_numpy(float), q.Hum4.to_numpy(float), q.elapsed_min.to_numpy(float))
            p7 = predict_ols(reg["m7_beta"], q.Temp7.to_numpy(float), q.Hum7.to_numpy(float), q.elapsed_min.to_numpy(float))
        else:
            p4 = np.full(len(q), reg["base4"], dtype=float)
            p7 = np.full(len(q), reg["base7"], dtype=float)
        m4 = q.Measure4.to_numpy(float)
        m7 = q.Measure7.to_numpy(float)
        if target_mode == "residual":
            t4, t7 = m4 - p4, m7 - p7
        elif target_mode == "log_measure":
            t4, t7 = np.log(np.maximum(m4, 1e-6)), np.log(np.maximum(m7, 1e-6))
        elif target_mode == "log_residual":
            t4 = np.log(np.maximum(m4, 1e-6)) - np.log(np.maximum(p4, 1e-6))
            t7 = np.log(np.maximum(m7, 1e-6)) - np.log(np.maximum(p7, 1e-6))
        else:
            raise ValueError(target_mode)
        env = np.stack([
            t4, t7,
            q.Temp4.to_numpy(float), q.Hum4.to_numpy(float),
            q.Temp7.to_numpy(float), q.Hum7.to_numpy(float),
            q.elapsed_min.to_numpy(float),
        ], axis=-1)
        x_env.append(env)
        x_det.append(make_detection(m4, m7, float(r.base4), float(r.base7)))
        abs_measure.append(np.stack([m4, m7], axis=-1))
        cond.append(env[:, 2:])
        pred.append(np.stack([p4, p7], axis=-1))
        rows.append({
            "id": int(i), "date": str(r.date), "phase": int(r.phase), "role": r.role,
            "label": int(r.label), "error": int(r.error), "start": str(r.start), "end": str(r.end),
            "row4": r.row4, "row7": r.row7, "base4": float(r.base4), "base7": float(r.base7),
        })

    np.savez_compressed(
        out / "dataset_env_conditioned.npz",
        x_env=np.asarray(x_env, dtype=np.float32),
        x_det=np.asarray(x_det, dtype=np.float32),
        abs_measure=np.asarray(abs_measure, dtype=np.float32),
        cond=np.asarray(cond, dtype=np.float32),
        pred=np.asarray(pred, dtype=np.float32),
    )
    save_json(out / "metadata_env_conditioned.json", rows)
    save_json(out / "environment_regressions.json", regressions)
    protocol = {
        "source_run": str(FIXED),
        "features": ["log(Measure4)", "log(Measure7)", "Temp4", "Hum4", "Temp7", "Hum7", "elapsed_min"] if target_mode == "log_measure" else FEATURES,
        "detection_features": DETECTION_FEATURES,
        "window_length": LENGTH,
        "target_channels": ["log(Measure4)", "log(Measure7)"] if target_mode == "log_measure" else (["log(Measure4/predicted_M4)", "log(Measure7/predicted_M7)"] if target_mode == "log_residual" else ["M4_resid_env", "M7_resid_env"]),
        "condition_channels": ["Temp4", "Hum4", "Temp7", "Hum7", "elapsed_min"],
        "residual": "per-date calibration OLS: Measure ~ 1 + Temp + Hum + elapsed_min" if residual_mode == "regression" else "per-date calibration baseline: Measure - base",
        "residual_mode": residual_mode,
        "target_mode": target_mode,
        "physical_constraint": "log target mode decodes generated targets with exp(), so raw generated Measure is strictly positive before projection" if target_mode in ("log_measure", "log_residual") else "residual mode needs projection to enforce M4>0 and M7>0",
        "mask": "Temp/Hum/elapsed observed for all steps; Measure residual observed for first half, generated for second half",
        "split": "inherits upstream_gpu_full_20260910 metadata roles and leave-one-date-out protocol",
    }
    save_json(out / "protocol_env_conditioned.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False), flush=True)


class EnvCSDI:
    def __init__(self, steps=30, hidden=32):
        p = ROOT / "upstream" / "CSDI"
        config = {
            "diffusion": {"layers": 2, "channels": hidden, "nheads": 4, "diffusion_embedding_dim": 64,
                          "beta_start": .0001, "beta_end": .5, "num_steps": steps, "schedule": "quad",
                          "is_linear": False, "seqlen": LENGTH},
            "model": {"is_unconditional": 0, "timeemb": 32, "featureemb": 8, "target_strategy": "random"},
        }
        load_module("diff_models", p / "diff_models.py")
        module = load_module("env_upstream_CSDI", p / "main_model.py")
        self.model = module.CSDI_base(len(FEATURES), config, "cpu")
        self.config = config
        self.module_file = inspect.getfile(type(self.model))

    def mask(self, x):
        obs = x.permute(0, 2, 1)
        mask = torch.ones_like(obs)
        mask[:, :2, LENGTH // 2:] = 0
        tp = torch.arange(LENGTH).float().repeat(len(x), 1)
        return obs, mask, self.model.get_side_info(tp, mask), tp

    def loss(self, x):
        obs, mask, side, _ = self.mask(x)
        full = torch.ones_like(obs)
        return self.model.calc_loss(obs, mask, full, side, 1)

    @torch.no_grad()
    def sample(self, anchors):
        self.model.eval()
        obs, mask, side, _ = self.mask(anchors)
        samples = self.model.impute(obs, mask, side, 1)[:, 0]
        return (mask * obs + (1 - mask) * samples).permute(0, 2, 1)


def classifier_features(x):
    return np.concatenate([x[:, -1], x.mean(1), x.std(1), x[:, -1] - x[:, 0]], axis=1)


def scores(y, prob, threshold):
    pred = prob >= threshold
    return {
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "Average Precision": float(average_precision_score(y, prob)) if np.asarray(y).sum() else None,
    }


def threshold_from_validation(y, p):
    candidates = np.unique(np.r_[p, .5, np.nextafter(p.max(), np.inf)])
    return float(max(candidates, key=lambda t: (f1_score(y, p >= t, zero_division=0), t)))


def train_classifier(name, xt, yt, xv, yv, xe, ye, seed, weight=None):
    if name == "RF":
        model = RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=2,
                                       random_state=seed, n_jobs=2, class_weight="balanced")
    else:
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=5,
                                               l2_regularization=1, random_state=seed)
    model.fit(classifier_features(xt), yt, sample_weight=weight)
    vp = model.predict_proba(classifier_features(xv))[:, 1]
    ep = model.predict_proba(classifier_features(xe))[:, 1]
    th = threshold_from_validation(yv, vp)
    return scores(ye, ep, th), scores(yv, vp, th), ep, th


def physical_bounds_by_date():
    audit = pd.read_csv(FIXED / "paired_row_audit.csv", dtype={"date": str})
    bounds = {}
    for date, g in audit[audit.agree].groupby("date"):
        bounds[str(date)] = {}
        for label in (0, 1):
            if label == 0:
                q = g[(g.error4 == 0) & g.role.isin(["calibration", "fit", "validation"])]
                margin = 0.05
            else:
                q = g[g.error4 != 0]
                margin = 0.20
            if q.empty:
                q = g
            m4_min, m4_max = float(q.Measure4.min()), float(q.Measure4.max())
            m7_min, m7_max = float(q.Measure7.min()), float(q.Measure7.max())
            bounds[str(date)][str(label)] = [
                max(m4_min * (1 - margin), 1e-6), m4_max * (1 + margin),
                max(m7_min * (1 - margin), 1e-6), m7_max * (1 + margin),
            ]
    return bounds


def project_generated(gen_env, pred, bases, target_mode="residual", source_dates=None, bounds=None, labels=None):
    if target_mode == "log_measure":
        raw_abs = np.exp(np.clip(gen_env[:, :, :2], -30, 30))
    elif target_mode == "log_residual":
        raw_abs = np.exp(np.clip(gen_env[:, :, :2], -30, 30) + np.log(np.maximum(pred, 1e-6)))
    else:
        raw_abs = gen_env[:, :, :2] + pred
    proj_abs = np.maximum(raw_abs, 1e-6)
    if source_dates is not None and bounds is not None:
        for i, date in enumerate(source_dates):
            label = str(int(labels[i])) if labels is not None else "0"
            lo4, hi4, lo7, hi7 = bounds[str(date)][label]
            proj_abs[i, :, 0] = np.clip(proj_abs[i, :, 0], lo4, hi4)
            proj_abs[i, :, 1] = np.clip(proj_abs[i, :, 1], lo7, hi7)
    det = np.empty((len(gen_env), LENGTH, 5), dtype=np.float32)
    for i in range(len(gen_env)):
        det[i] = make_detection(proj_abs[i, :, 0], proj_abs[i, :, 1], bases[i, 0], bases[i, 1])
    return raw_abs.astype(np.float32), proj_abs.astype(np.float32), det


def run(args):
    started = time.time()
    seed_all(args.seed)
    out = args.output
    run_dir = out / "CSDI" / f"seed{args.seed}" / args.test_date
    run_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(out / "dataset_env_conditioned.npz")
    protocol = json.loads((out / "protocol_env_conditioned.json").read_text(encoding="utf-8"))
    target_mode = protocol.get("target_mode", "residual")
    x_env = data["x_env"]
    x_det = data["x_det"]
    pred_env = data["pred"]
    meta = pd.read_json(out / "metadata_env_conditioned.json", dtype={"date": str}).reset_index(drop=True)
    y = meta.label.to_numpy(int)
    fit = np.flatnonzero((meta.date != args.test_date) & (meta.role == "fit"))
    val = np.flatnonzero((meta.date != args.test_date) & (meta.role == "validation"))
    test = np.flatnonzero(meta.date == args.test_date)
    for name, ids in [("fit", fit), ("validation", val), ("test", test)]:
        if len(np.unique(y[ids])) != 2:
            raise ValueError(f"{name} must have both classes")

    scaler = MinMaxScaler(feature_range=(-.8, .8)).fit(x_env[fit].reshape(-1, len(FEATURES)))
    z = scaler.transform(x_env.reshape(-1, len(FEATURES))).reshape(x_env.shape).astype(np.float32)
    generated, generated_y, anchor_ids, histories, source_files = [], [], [], [], {}

    for label in (0, 1):
        ids = fit[y[fit] == label]
        model = EnvCSDI(args.steps, args.hidden)
        source_files[model.module_file] = sha(model.module_file)
        optimizer = torch.optim.Adam(model.model.parameters(), lr=.001)
        inp = torch.from_numpy(z[ids])
        hist = []
        for epoch in range(args.epochs):
            model.model.train()
            order = torch.randperm(len(inp))
            losses = []
            for start in range(0, len(inp), args.batch):
                selected = order[start:start + args.batch]
                loss = model.loss(inp[selected])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1)
                optimizer.step()
                losses.append(float(loss.detach()))
            hist.append(float(np.mean(losses)))
            if epoch in (0, args.epochs - 1) or (epoch + 1) % 20 == 0:
                print(f"CSDI env {args.test_date} seed={args.seed} class={label} epoch={epoch+1} loss={hist[-1]:.6f}", flush=True)
        torch.save({"state_dict": model.model.state_dict(), "config": model.config, "scaler_min": scaler.min_,
                    "scaler_scale": scaler.scale_, "fit_ids": ids, "features": FEATURES}, run_dir / f"class{label}.pt")
        histories.append({"label": label, "n_train": int(len(ids)), "loss": hist,
                          "parameters": int(sum(p.numel() for p in model.model.parameters()))})
        # In target_date mode, keep the model trained only on the other dates,
        # but replace the source anchor/environment trajectory with real windows
        # from the held-out date. This is a physical validation mode, not a
        # leakage-free downstream classifier benchmark.
        anchor_pool = test[y[test] == label] if args.condition_source == "target_date" else ids
        count = max(args.generate // 2, 100 * len(fit) // 2)
        rng = np.random.default_rng(args.seed + label)
        chosen = rng.choice(anchor_pool, count, replace=True)
        chunks = []
        for start in range(0, count, args.sample_batch):
            a = chosen[start:start + args.sample_batch]
            s = model.sample(torch.from_numpy(z[a])).numpy()
            chunks.append(s)
        gen = np.concatenate(chunks)
        generated.append(scaler.inverse_transform(gen.reshape(-1, len(FEATURES))).reshape(gen.shape).astype(np.float32))
        generated_y.extend([label] * len(gen))
        anchor_ids.extend(chosen.tolist())

    synthetic_env = np.concatenate(generated)
    sy = np.asarray(generated_y, dtype=int)
    anchors = np.asarray(anchor_ids, dtype=int)
    bases = meta.loc[anchors, ["base4", "base7"]].to_numpy(float)
    source_dates = meta.loc[anchors, "date"].astype(str).to_numpy()
    bounds = physical_bounds_by_date()
    raw_abs, projected_abs, projected_det = project_generated(synthetic_env, pred_env[anchors], bases, target_mode, source_dates, bounds, sy)
    raw_positive = ((raw_abs[:, :, 0] > 0) & (raw_abs[:, :, 1] > 0)).all(axis=1)

    np.savez_compressed(run_dir / "synthetic_env_conditioned.npz", x_env=synthetic_env, raw_abs=raw_abs,
                        projected_abs=projected_abs, projected_det=projected_det, label=sy,
                        anchor_ids=anchors, raw_positive=raw_positive)

    rows = []
    rng = np.random.default_rng(args.seed)
    order = np.ravel(np.column_stack([rng.permutation(np.flatnonzero(sy == 0)), rng.permutation(np.flatnonzero(sy == 1))]))
    for clf in ("RF", "HGB"):
        ev, vv, prob, threshold = train_classifier(clf, x_det[fit], y[fit], x_det[val], y[val], x_det[test], y[test], args.seed)
        rows.append({"method": "Real-only", "classifier": clf, "regime": "real", "n_synthetic": 0,
                     "metrics": ev, "validation": vv, "threshold": threshold})
        for regime, n in (("1x", len(fit)), ("large", len(projected_det))):
            sel = order[:n]
            xx = np.concatenate([x_det[fit], projected_det[sel]])
            yy = np.r_[y[fit], sy[sel]]
            weights = np.r_[np.ones(len(fit)), np.full(len(sel), len(fit) / len(sel))]
            ev, vv, prob, threshold = train_classifier(clf, xx, yy, x_det[val], y[val], x_det[test], y[test], args.seed, weights)
            rows.append({"method": "EnvConditioned-CSDI", "classifier": clf, "regime": f"projected_{regime}",
                         "n_synthetic": int(len(sel)), "metrics": ev, "validation": vv, "threshold": threshold})

    result = {
        "method": "EnvConditioned-CSDI", "seed": args.seed, "test_date": args.test_date,
        "features": protocol.get("features", FEATURES), "detection_features": DETECTION_FEATURES,
        "condition_channels": FEATURES[2:], "target_channels": FEATURES[:2],
        "residual": protocol.get("residual"), "residual_mode": protocol.get("residual_mode"),
        "target_mode": target_mode, "physical_constraint": protocol.get("physical_constraint"),
        "condition_source": args.condition_source,
        "warning": "target_date mode uses held-out Measure first-half anchors for physical continuation validation; do not use its classifier metrics as leakage-free test results",
        "mask": "conditions all observed; target first half observed and second half generated",
        "adapted_parameters": {"seq_len": LENGTH, "feature_dim": len(FEATURES), "hidden": args.hidden,
                               "batch": args.batch, "diffusion_steps": args.steps, "epochs": args.epochs,
                               "device": "cpu"},
        "upstream_sources": source_files, "fit_ids": fit.tolist(), "val_ids": val.tolist(), "test_ids": test.tolist(),
        "generated_windows": int(len(synthetic_env)), "raw_positive_windows": int(raw_positive.sum()),
        "raw_nonpositive_row_pct": float(((raw_abs[:, :, 0] <= 0) | (raw_abs[:, :, 1] <= 0)).mean() * 100),
        "projection_bounds_by_source_date": bounds,
        "training": histories, "results": rows, "elapsed_seconds": time.time() - started,
    }
    save_json(run_dir / "result.json", result)
    print(f"COMPLETED env CSDI {args.test_date} seed {args.seed}: {len(synthetic_env)} generated", flush=True)


def make_reports(out):
    meta = pd.read_json(out / "metadata_env_conditioned.json", dtype={"date": str}).reset_index(drop=True)
    audit_src = pd.read_csv(FIXED / "paired_row_audit.csv", dtype={"date": str})
    audit_src["dt"] = pd.to_datetime(audit_src["dt"])
    by_row4 = {(str(r.date), int(r.row4)): r for r in audit_src.itertuples(index=False)}
    rows, audit_rows, metric_rows = [], [], []
    rng = np.random.default_rng(20260916)
    for date in DATES:
        g = audit_src[audit_src.date == date].sort_values("dt")
        for r in g.itertuples(index=False):
            rows.append({"date": date, "source": "real", "time": r.time4, "Measure4": r.Measure4,
                         "Measure7": r.Measure7, "Temp4": r.Temp4, "Hum4": r.Hum4, "Temp7": r.Temp7,
                         "Hum7": r.Hum7, "error": int(r.error4), "condition_source": "real", "mode": "real"})
        syn_path = out / "CSDI" / "seed42" / date / "synthetic_env_conditioned.npz"
        if not syn_path.exists():
            continue
        syn = np.load(syn_path)
        labels, anchors = syn["label"], syn["anchor_ids"]
        raw_abs, projected_abs = syn["raw_abs"], syn["projected_abs"]
        pick = rng.choice(len(labels), min(160, len(labels)), replace=False)
        for mode, arr in (("raw", raw_abs), ("projected", projected_abs)):
            for i in pick:
                anchor = meta.iloc[int(anchors[i])]
                for step, row4 in enumerate(anchor.row4):
                    env = by_row4[(str(anchor.date), int(row4))]
                    rows.append({"date": date, "source": "EnvCSDI", "time": "", "Measure4": arr[i, step, 0],
                                 "Measure7": arr[i, step, 1], "Temp4": env.Temp4, "Hum4": env.Hum4,
                                 "Temp7": env.Temp7, "Hum7": env.Hum7, "error": int(labels[i]),
                                 "condition_source": str(anchor.date), "mode": mode})
        for label in (0, 1):
            ids = labels == label
            for mode, arr in (("raw", raw_abs), ("projected", projected_abs)):
                audit_rows.append({"test_date": date, "label": label, "mode": mode, "n_windows": int(ids.sum()),
                                   "M4_nonpositive_row_pct": float((arr[ids, :, 0] <= 0).mean() * 100),
                                   "M7_nonpositive_row_pct": float((arr[ids, :, 1] <= 0).mean() * 100),
                                   "M4_min": float(arr[ids, :, 0].min()), "M7_min": float(arr[ids, :, 1].min()),
                                   "M4_max": float(arr[ids, :, 0].max()), "M7_max": float(arr[ids, :, 1].max())})
        result = json.loads((out / "CSDI" / "seed42" / date / "result.json").read_text(encoding="utf-8"))
        for item in result["results"]:
            metric_rows.append({"test_date": date, "classifier": item["classifier"], "method": item["method"],
                                "regime": item["regime"], "n_synthetic": item["n_synthetic"], **item["metrics"]})

    df = pd.DataFrame(rows)
    numeric = ["Measure4", "Measure7", "Temp4", "Hum4", "Temp7", "Hum7"]
    df[numeric] = df[numeric].round(4)
    xlsx = out / "real_vs_env_conditioned_generated.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        for date in DATES:
            part = df[df.date == date].drop(columns=["date"])
            part.to_excel(xl, sheet_name=date, index=False)
            ws = xl.sheets[date]
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")
            fill = PatternFill("solid", fgColor="E2F0D9")
            for row in range(2, ws.max_row + 1):
                if ws.cell(row, 1).value != "real":
                    for col in range(1, ws.max_column + 1):
                        ws.cell(row, col).fill = fill
                for col in range(1, ws.max_column + 1):
                    ws.cell(row, col).alignment = Alignment(horizontal="center")
            ws.freeze_panes = "A2"
        pd.DataFrame(audit_rows).round(4).to_excel(xl, sheet_name="physical_audit", index=False)
        pd.DataFrame(metric_rows).round(4).to_excel(xl, sheet_name="metrics", index=False)
    df.to_csv(out / "real_vs_env_conditioned_generated.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).round(4).to_csv(out / "env_conditioned_physical_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).round(4).to_csv(out / "env_conditioned_metrics.csv", index=False, encoding="utf-8-sig")
    print("wrote", xlsx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--method", default="CSDI", choices=["CSDI"])
    ap.add_argument("--test-date", default="250718")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--generate", type=int, default=2000)
    ap.add_argument("--sample-batch", type=int, default=256)
    ap.add_argument("--residual-mode", choices=["regression", "baseline"], default="regression")
    ap.add_argument("--target-mode", choices=["residual", "log_measure", "log_residual"], default="residual")
    ap.add_argument("--condition-source", choices=["fit_dates", "target_date"], default="fit_dates")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(args.output, args.residual_mode, args.target_mode)
    elif args.report:
        make_reports(args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()

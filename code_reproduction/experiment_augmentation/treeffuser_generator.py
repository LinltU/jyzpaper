"""Treeffuser generator: fixed-version upstream package (blei-lab/treeffuser).

Treeffuser = tree-based diffusion (gradient-boosted-tree score model) for tabular
data (arXiv 2406.07658). Our 4x5 windows flatten to 20-dim tabular rows, so this is
a fixed-version upstream run (imports the pinned repo in ../upstream/Treeffuser),
not a mechanism port. Evaluation mirrors all_papers_extended.run_generator exactly
so numbers are comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from upstream_large_augmentation import (
    FEATURES,
    LENGTH,
    ROOT,
    prepare,
    save_json,
    scores,
    seed_all,
    sha,
    threshold_from_validation,
    train_classifier,
)

HERE = Path(__file__).resolve().parent
TREEFFUSER_SRC = ROOT / "upstream" / "Treeffuser" / "src"
sys.path.insert(0, str(TREEFFUSER_SRC))
from treeffuser import Treeffuser  # noqa: E402

METHOD = "Treeffuser"


def load_split(out_dir, test_date):
    x = np.load(out_dir / "dataset.npz")["x"]
    meta = pd.read_json(out_dir / "metadata.json", dtype={"date": str})
    y = meta.label.to_numpy()
    fit = np.flatnonzero((meta.date != test_date) & (meta.role == "fit"))
    val = np.flatnonzero((meta.date != test_date) & (meta.role == "validation"))
    test = np.flatnonzero(meta.date == test_date)
    return x, y, fit, val, test, meta


def scale_data(train, arrays):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-.8, .8)).fit(train.reshape(-1, 5))
    return scaler, [scaler.transform(a.reshape(-1, 5)).reshape(a.shape).astype(np.float32) for a in arrays]


def run(args):
    out = args.output
    run_dir = out / "generators" / METHOD / f"seed{args.seed}" / args.test_date
    if (run_dir / "result.json").exists():
        print(f"skip {run_dir}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed_all(args.seed)
    x, y, fit, val, test, meta = load_split(out, args.test_date)
    scaler, (z,) = scale_data(x[fit], [x])
    generated, generated_y, anchor_ids = [], [], []
    train_audit = []
    for label in (0, 1):
        ids = fit[y[fit] == label]
        flat = z[ids].reshape(len(ids), LENGTH * len(FEATURES))
        model = Treeffuser(sde_initialize_from_data=True, seed=args.seed)
        model.fit(np.zeros((len(ids), 1)), flat)
        count = args.generate // 2
        aseed = args.seed + label
        samp = model.sample(np.zeros((count, 1)), n_samples=1, n_steps=20, seed=aseed)
        syn = samp[0].reshape(count, LENGTH, len(FEATURES)).astype(np.float32)
        generated.append(scaler.inverse_transform(syn.reshape(-1, 5)).reshape(syn.shape).astype(np.float32))
        generated_y.extend([label] * count)
        anchor_ids.extend(np.arange(count) % len(ids) + 0)
        train_audit.append({"label": label, "n_train": int(len(ids))})
        print(f"Treeffuser class={label} generated {count}", flush=True)
    synthetic = np.concatenate(generated)
    sy = np.asarray(generated_y)
    # anchors: sample per-class rows for physical projection baseline
    anchor_ids = []
    for label in (0, 1):
        rng = np.random.default_rng(args.seed + label)
        ids = fit[y[fit] == label]
        anchor_ids.extend(rng.choice(ids, args.generate // 2, replace=True).tolist())
    anchors = np.asarray(anchor_ids)
    base = meta[["base4", "base7"]].to_numpy()[anchors]
    m4 = np.maximum(synthetic[:, :, 0] + base[:, None, 0], 1e-6)
    m7 = np.maximum(synthetic[:, :, 1] + base[:, None, 1], 1e-6)
    positive = ((synthetic[:, :, 0] + base[:, None, 0] > 0) & (synthetic[:, :, 1] + base[:, None, 1] > 0)).all(axis=1)
    corrected = synthetic.copy()
    corrected[:, :, 0] = m4 - base[:, None, 0]
    corrected[:, :, 1] = m7 - base[:, None, 1]
    corrected[:, :, 2] = m4 - m7
    corrected[:, :, 3] = m4 / m7
    corrected[:, :, 4] = np.log(m4 / m7)
    np.savez_compressed(run_dir / "synthetic.npz",
                        raw=synthetic.astype(np.float32), projected=corrected.astype(np.float32),
                        label=sy.astype(np.int64), anchor_ids=anchors.astype(np.int64),
                        positive=positive.astype(bool))
    rows = []
    rng = np.random.default_rng(args.seed)
    order = np.ravel(np.column_stack([rng.permutation(np.flatnonzero(sy == 0)), rng.permutation(np.flatnonzero(sy == 1))]))
    for clf in ("RF", "HGB"):
        _, ev, vv, prob, threshold = train_classifier(clf, x[fit], y[fit], x[val], y[val], x[test], y[test], args.seed)
        rows.append({"method": "Real-only", "classifier": clf, "regime": "real", "n_synthetic": 0,
                     "metrics": ev, "validation": vv, "threshold": threshold})
        regimes = [("1x", len(fit)), ("10x", 10 * len(fit)), ("100x", 100 * len(fit)), ("large", len(synthetic))]
        for mode, pool in [("raw", synthetic), ("projected", corrected)]:
            valid_order = order if mode == "raw" else order[positive[order]]
            for regime, n in regimes:
                sel = valid_order[:n]
                if not len(sel):
                    continue
                xx = np.concatenate([x[fit], pool[sel]])
                yy = np.r_[y[fit], sy[sel]]
                weights = np.r_[np.ones(len(fit)), np.full(len(sel), len(fit) / len(sel))]
                _, ev, vv, prob, threshold = train_classifier(clf, xx, yy, x[val], y[val], x[test], y[test], args.seed, weights)
                rows.append({"method": METHOD, "classifier": clf, "regime": f"{mode}_{regime}",
                             "n_synthetic": len(sel), "metrics": ev, "validation": vv, "threshold": threshold})
    result = {
        "method": METHOD, "seed": args.seed, "test_date": args.test_date, "status": "completed",
        "official_code_used": True, "implementation": "fixed-version upstream import",
        "upstream_repo": "https://github.com/blei-lab/treeffuser",
        "upstream_model_file": str(TREEFFUSER_SRC / "treeffuser" / "treeffuser.py"),
        "upstream_model_sha256": sha(TREEFFUSER_SRC / "treeffuser" / "treeffuser.py"),
        "mechanism": "tree-based diffusion (LightGBM score model) over flattened 20-dim window",
        "adapted_parameters": {"seq_len": LENGTH, "feature_dim": len(FEATURES), "flatten_dim": LENGTH * len(FEATURES),
                               "diffusion_steps": 20, "sde_initialize_from_data": True},
        "features": FEATURES, "generated_windows": len(synthetic), "positive_windows": int(positive.sum()),
        "generation_std": synthetic.std(axis=(0, 1)).tolist(), "real_std": x[fit].std(axis=(0, 1)).tolist(),
        "training": train_audit, "results": rows, "elapsed_seconds": time.time() - started,
        "input_sha256": sha(ROOT.parent / "dataset" / "experiment.xlsx"),
    }
    save_json(run_dir / "result.json", result)
    print(f"COMPLETED gen {METHOD} {args.test_date} seed {args.seed}: {len(synthetic)} generated, {len(rows)} evals, {time.time()-started:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-date", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generate", type=int, default=20000)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()

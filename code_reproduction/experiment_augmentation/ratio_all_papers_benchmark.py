"""Train paper generator schemes on the updated workbook at train:test ratios.

The split is chronological within each (date, error-stage) group. The first
portion is training/validation and the tail is held out for testing. All methods
use the same paired 4-step, 5-feature representation and downstream RF/HGB
evaluation. This runner records the raw generated regime without clipping.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
WORKBOOK = HERE.parent.parent / "experiment.xlsx"
UPSTREAM = HERE.parent / "runs" / "upstream_gpu_full_20260918"
# Names follow the user's test:train convention. Values are the train fraction.
RATIOS = {"2to8": 0.8, "3to7": 0.7, "4to6": 0.6, "5to5": 0.5}
UPSTREAM_METHODS = ["CSDI", "MTSCI", "IMDiffusion", "Diffusion-TS", "D3R"]
MECHANISM_METHODS = ["TimeGrad", "NsDiff", "FIDE", "PaD-TS", "WaveStitch", "TabDDPM", "CCDM", "DiffPLF", "TSDE", "SSSD", "CoSTI", "SSD-TS", "MCD-TSF"]
METHODS = UPSTREAM_METHODS + MECHANISM_METHODS
FEATURES = ["M4_residual", "M7_residual", "M4-M7", "M4/M7", "log(M4/M7)"]
LENGTH = 4


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def build_windows(stable_only=False):
    sheets = pd.read_excel(WORKBOOK, sheet_name=None)
    rows, meta, audits = [], [], []
    for date in sorted({s[:6] for s in sheets}):
        s4 = next(s for s in sheets if s[:6] == date and "(4)" in s)
        s7 = next(s for s in sheets if s[:6] == date and "(7)" in s)
        a = sheets[s4].copy(); b = sheets[s7].copy()
        a["dt"] = pd.to_datetime(a.time.astype(str).str.replace("_", " ", regex=False)); b["dt"] = pd.to_datetime(b.time.astype(str).str.replace("_", " ", regex=False))
        a = a.sort_values("dt").reset_index(drop=True); b = b.sort_values("dt").reset_index(drop=True)
        a["row4"] = np.arange(len(a)) + 2; b["row7"] = np.arange(len(b)) + 2
        q = pd.merge_asof(a, b, on="dt", direction="nearest", tolerance=pd.Timedelta("40s"), suffixes=("4", "7")).dropna(subset=["Measure7"])
        q = q[q.error4 == q.error7].copy().reset_index(drop=True)
        q["phase"] = ((q.error4 != q.error4.shift()) | (q.dt.diff().dt.total_seconds() > 180)).cumsum()
        normal = q[q.error4 == 0]
        stable_normal = normal
        calibration = normal.head(8)
        if stable_only and len(normal):
            tail = normal.tail(40)
            terminal = tail[["Measure4", "Measure7"]].tail(10).median().to_numpy()
            relative = np.abs(tail[["Measure4", "Measure7"]].to_numpy() / np.maximum(terminal, 1e-6) - 1).max(axis=1)
            bad = np.flatnonzero(relative > .15)
            stable_normal = tail.iloc[(bad[-1] + 1 if len(bad) else 0):]
            calibration = stable_normal.head(8)
        base = calibration[["Measure4", "Measure7"]].median().to_numpy() if len(calibration) >= 8 else normal.head(8)[["Measure4", "Measure7"]].median().to_numpy()
        elapsed = (q.dt - q.dt.min()).dt.total_seconds().to_numpy() / 60
        m4, m7 = q.Measure4.to_numpy(float), q.Measure7.to_numpy(float)
        q[FEATURES[0]] = m4 - base[0]; q[FEATURES[1]] = m7 - base[1]
        q[FEATURES[2]] = m4 - m7; q[FEATURES[3]] = (m4 + 1e-4) / (m7 + 1e-4); q[FEATURES[4]] = np.log((m4 + 1e-4) / (m7 + 1e-4))
        for phase, g in q.groupby("phase"):
            if stable_only:
                if int(g.error4.iloc[0]) == 0:
                    valid = g[g.index.isin(stable_normal.iloc[8:].index)]
                else:
                    valid = g.iloc[2:]
            else:
                valid = g
            for start in range(max(0, len(valid) - LENGTH + 1)):
                z = valid.iloc[start:start + LENGTH]
                rows.append(z[FEATURES].to_numpy(np.float32))
                meta.append({"date": date, "phase": int(phase), "error": int(z.error4.iloc[0]), "label": int(z.error4.iloc[0] != 0), "base4": float(base[0]), "base7": float(base[1]), "rows4": z.row4.astype(int).tolist()})
        audits.append({"date": date, "paired_rows": len(q), "base4": float(base[0]), "base7": float(base[1]), "errors": q.error4.value_counts().to_dict(), "stable_stage_filter": bool(stable_only), "stable_normal_rows": int(len(stable_normal)), "calibration_rows": int(len(calibration))})
    return np.asarray(rows, np.float32), pd.DataFrame(meta), audits


def split_ratio(meta, frac):
    fit, val, test = [], [], []
    for _, g in meta.groupby(["date", "phase"], sort=False):
        ids = g.index.to_numpy(); cut = max(1, min(len(ids) - 1, int(len(ids) * frac))) if len(ids) > 1 else 1
        train_ids, test_ids = ids[:cut], ids[cut:]
        if len(train_ids) > 2:
            vcut = max(1, int(len(train_ids) * .8)); fit.extend(train_ids[:vcut]); val.extend(train_ids[vcut:])
        else: fit.extend(train_ids)
        test.extend(test_ids)
    return np.array(fit, int), np.array(val, int), np.array(test, int)


def clf_features(x): return np.concatenate([x[:, -1], x.mean(1), x.std(1), x[:, -1] - x[:, 0]], axis=1)


def threshold(y, p):
    cs = np.unique(np.r_[p, .5]); return float(max(cs, key=lambda t: (f1_score(y, p >= t, zero_division=0), t)))


def evaluate(xfit, yfit, xval, yval, xtest, ytest, seed):
    out = []
    for name in ("RF", "HGB"):
        if name == "RF": model = RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=2)
        else: model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=5, l2_regularization=1, random_state=seed)
        model.fit(clf_features(xfit), yfit)
        pv = model.predict_proba(clf_features(xval))[:, 1] if len(xval) else model.predict_proba(clf_features(xfit))[:, 1]
        pe = model.predict_proba(clf_features(xtest))[:, 1]
        th = threshold(yval if len(xval) else yfit, pv)
        pred = pe >= th
        out.append({"classifier": name, "Precision": precision_score(ytest, pred, zero_division=0), "Recall": recall_score(ytest, pred, zero_division=0), "F1": f1_score(ytest, pred, zero_division=0), "ROC-AUC": roc_auc_score(ytest, pe) if len(np.unique(ytest)) == 2 else np.nan, "Average Precision": average_precision_score(ytest, pe) if ytest.sum() else np.nan})
    return out


def main_run(args):
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    x, meta, audits = build_windows(args.stable_only); meta.to_json(out / "metadata.json", orient="records", force_ascii=False, indent=2)
    (out / "protocol.json").write_text(json.dumps({"source": str(WORKBOOK), "ratio_definition": "test:train", "ratios": RATIOS, "stable_stage_filter": bool(args.stable_only), "stable_stage_rule": "stable normal suffix within 15% of terminal 10-point median; discard first 2 matched points of abnormal phases", "features": FEATURES, "window_length": LENGTH, "methods": METHODS, "epochs": args.epochs, "seed": args.seed, "audits": audits}, ensure_ascii=False, indent=2), encoding="utf-8")
    all_rows=[]; train_rows=[]
    mech = load_mod("ratio_mech", HERE / "all_papers_extended.py")
    upstream = load_mod("ratio_upstream", HERE / "upstream_large_augmentation.py")
    for ratio_name, frac in RATIOS.items():
        fit, val, test = split_ratio(meta, frac)
        if not len(test) or len(np.unique(meta.iloc[test].label)) < 2: continue
        for method in METHODS:
            for seed in [args.seed]:
                seed_all(seed); scaler=MinMaxScaler(feature_range=(-.8,.8)).fit(x[fit].reshape(-1,5)); z=scaler.transform(x.reshape(-1,5)).reshape(x.shape).astype(np.float32)
                generated=[]; gen_labels=[]; generated_sources=[]
                for label in (0,1):
                    ids=fit[meta.iloc[fit].label.to_numpy()==label]
                    if not len(ids): continue
                    if method in UPSTREAM_METHODS:
                        model=upstream.Upstream(method,args.steps,args.hidden); opt=torch.optim.Adam(model.model.parameters(),lr=.001); inp=torch.from_numpy(z[ids])
                        losses=[]
                        for ep in range(args.epochs):
                            order=torch.randperm(len(inp)); ls=[]
                            for st in range(0,len(inp),32):
                                sel=order[st:st+32]; loss=model.loss(inp[sel]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.model.parameters(),1); opt.step(); ls.append(float(loss.detach()))
                            losses.append(float(np.mean(ls)))
                        n=max(20, len(ids)*args.multiplier); chosen=np.random.default_rng(seed+label).choice(ids,n,replace=True); chunks=[]
                        for st in range(0,n,args.batch): chunks.append(model.sample(torch.from_numpy(z[chosen[st:st+args.batch]])).numpy())
                        syn=scaler.inverse_transform(np.concatenate(chunks).reshape(-1,5)).reshape(-1,LENGTH,5)
                    else:
                        model=mech.MechanismGenerator(method,args.steps,args.hidden,seed); opt=torch.optim.Adam(model.ddpm.parameters(),lr=.001); inp=torch.from_numpy(z[ids]); losses=[]
                        for ep in range(args.epochs):
                            order=torch.randperm(len(inp)); ls=[]
                            for st in range(0,len(inp),32):
                                sel=order[st:st+32]; loss=model.loss(inp[sel],label); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.ddpm.parameters(),1); opt.step(); ls.append(float(loss.detach()))
                            losses.append(float(np.mean(ls)))
                        n=max(20,len(ids)*args.multiplier); chosen=np.random.default_rng(seed+label).choice(ids,n,replace=True); syn=model.sample(torch.from_numpy(z[chosen]), label, n).numpy(); syn=scaler.inverse_transform(syn.reshape(-1,5)).reshape(-1,LENGTH,5)
                    generated.append(syn.astype(np.float32)); gen_labels.extend([label]*len(syn)); generated_sources.extend(chosen.tolist()); train_rows.append({"ratio":ratio_name,"method":method,"label":label,"n_train":len(ids),"n_generated":len(syn),"loss":losses[-1]})
                synthetic=np.concatenate(generated); sy=np.asarray(gen_labels); source_ids=np.asarray(generated_sources, dtype=np.int64)
                safe_method=method.replace("/", "_").replace("-", "_")
                np.savez_compressed(out / f"raw_{ratio_name}_{safe_method}.npz", windows=synthetic, labels=sy, source_ids=source_ids)
                for mode,pool in (("raw",synthetic),):
                    for clfrow in evaluate(x[fit],meta.iloc[fit].label.to_numpy(),x[val],meta.iloc[val].label.to_numpy(),x[test],meta.iloc[test].label.to_numpy(),seed):
                        clfrow.update({"ratio":ratio_name,"method":"Real-only","mode":"real","seed":seed,"n_synthetic":0}); all_rows.append(clfrow)
                    xx=np.concatenate([x[fit],pool]); yy=np.r_[meta.iloc[fit].label.to_numpy(),sy]
                    for clfrow in evaluate(xx,yy,x[val],meta.iloc[val].label.to_numpy(),x[test],meta.iloc[test].label.to_numpy(),seed):
                        clfrow.update({"ratio":ratio_name,"method":method,"mode":mode,"seed":seed,"n_synthetic":len(pool)}); all_rows.append(clfrow)
                print(f"completed ratio={ratio_name} method={method} train={len(fit)} test={len(test)}",flush=True)
    pd.DataFrame(all_rows).to_csv(out/"all_results.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(train_rows).to_csv(out/"training_audit.csv",index=False,encoding="utf-8-sig")
    summary=pd.DataFrame(all_rows).groupby(["ratio","method","mode","classifier"]).agg(runs=("F1","size"),**{m:(m,"mean") for m in ["Precision","Recall","F1","ROC-AUC","Average Precision"]}).reset_index().sort_values(["ratio","Average Precision"],ascending=[True,False]); summary.to_csv(out/"summary.csv",index=False,encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=UPSTREAM); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--steps',type=int,default=20); ap.add_argument('--hidden',type=int,default=32); ap.add_argument('--multiplier',type=int,default=2); ap.add_argument('--batch',type=int,default=128); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--stable-only',action='store_true'); main_run(ap.parse_args())

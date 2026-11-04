"""Conditional diffusion augmentation for the 9-piece, 2-device experiment.

This is a local experiment-data adapter, not a claim that the public paper
repositories accept this workbook directly.  It uses a conditional DDPM over
short paired windows and an NsDiff-style local normalization variant.

The workbook has six sheets: three experiments and two synchronized devices
(piece 4 and piece 7) per experiment.  We pair rows by within-experiment
order, keep only windows within one ``error`` run, and condition generation on
the error position and experiment date.  ``error`` is a training condition,
not an input available to an online detector.
"""
from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score


FEATURES_DEVICE_ENV = [
    "M4", "M7", "Temp4", "Hum4", "Temp7", "Hum7",
    "Mdiff", "Mratio", "logratio", "Tdiff", "Hdiff", "elapsed_min",
]
FEATURES_SHARED_ENV = [
    "M4", "M7", "Temp_shared", "Hum_shared",
    "Mdiff", "Mratio", "logratio", "elapsed_min",
]
ERRORS = [0, 1, 2, 3, 5, 6, 8, 9]


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def load_windows(path: Path, length: int, features):
    if path.is_dir():
        sheets = {p.stem: pd.read_csv(p) for p in path.glob("*.csv")}
    else:
        sheets = pd.read_excel(path, sheet_name=None)
    groups = {}
    for name, df in sheets.items():
        date = name[:6]
        pos = int(name.split("(")[1][0])
        groups.setdefault(date, {})[pos] = df.reset_index(drop=True)
    rows = []
    meta = []
    for date, pair in sorted(groups.items()):
        a, b = pair[4].copy(), pair[7].copy()
        a["dt"] = pd.to_datetime(a.time, format="%Y-%m-%d_%H:%M:%S")
        b["dt"] = pd.to_datetime(b.time, format="%Y-%m-%d_%H:%M:%S")
        # The two devices have different sampling counts. Align by timestamp,
        # not by row number, so later error stages are not discarded.
        a = a.sort_values("dt"); b = b.sort_values("dt")
        # Sampling clocks differ by a few seconds and occasionally one device
        # misses a record; a 40-second nearest-time tolerance keeps the same
        # physical stage without requiring identical timestamps.
        b = pd.merge_asof(a[["dt"]], b, on="dt", direction="nearest", tolerance=pd.Timedelta("40s"))
        a = a.reset_index(drop=True); b = b.reset_index(drop=True)
        keep = b.Measure.notna().values
        a, b = a.loc[keep].reset_index(drop=True), b.loc[keep].reset_index(drop=True)
        t0 = a.dt.iloc[0]
        a["elapsed_min"] = (a.dt - t0).dt.total_seconds()/60
        b["elapsed_min"] = (b.dt - t0).dt.total_seconds()/60
        a["M4"], a["Temp4"], a["Hum4"] = a.Measure, a.Temp, a.Hum
        b["M7"], b["Temp7"], b["Hum7"] = b.Measure, b.Temp, b.Hum
        x = pd.DataFrame({
            "M4": a.M4, "M7": b.M7, "Temp4": a.Temp4, "Hum4": a.Hum4,
            "Temp7": b.Temp7, "Hum7": b.Hum7,
            "Mdiff": a.M4-b.M7,
            "Mratio": (a.M4+1e-3)/(b.M7+1e-3),
            "logratio": np.log((a.M4+1e-3)/(b.M7+1e-3)),
            "Tdiff": a.Temp4-b.Temp7, "Hdiff": a.Hum4-b.Hum7,
            "Temp_shared": (a.Temp4+b.Temp7)/2.0,
            "Hum_shared": (a.Hum4+b.Hum7)/2.0,
            "elapsed_min": a.elapsed_min,
            "error": a.error.astype(int).values,
            "date": date,
        })
        # Build windows inside a single error run. This prevents a synthetic
        # sample from mixing two physical re-powering/error stages.
        start = 0
        for i in range(1, len(x)+1):
            if i == len(x) or int(x.error.iloc[i]) != int(x.error.iloc[start]):
                run = x.iloc[start:i]
                for j in range(0, max(0, len(run)-length+1)):
                    z = run.iloc[j:j+length]
                    rows.append(z[features].to_numpy(dtype=np.float32))
                    meta.append({"date": date, "error": int(z.error.iloc[0]),
                                 "start_index": int(start+j),
                                 "start_elapsed_min": float(z.elapsed_min.iloc[0])})
                start = i
    return np.stack(rows), pd.DataFrame(meta)


class CondDDPM(nn.Module):
    def __init__(self, dim, cond_dim, hidden=256, steps=20):
        super().__init__(); self.steps = steps
        self.net = nn.Sequential(nn.Linear(dim+cond_dim+32, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, dim))
        beta = torch.linspace(1e-4, .02, steps); self.register_buffer("beta", beta)
        self.register_buffer("alpha", 1-beta); self.register_buffer("abar", torch.cumprod(1-beta, 0))
    def emb(self, t):
        half=16; f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/(half-1))
        z=t[:,None].float()*f[None,:]; return torch.cat([z.sin(),z.cos()],1)
    def forward(self, z, t, c): return self.net(torch.cat([z,c,self.emb(t)],1))


def one_hot(meta, dates, label_mode="binary"):
    c=[]
    for _, r in meta.iterrows():
        if label_mode == "binary":
            e=[1.0 if int(r.error)!=0 else 0.0]
        else:
            e=[1.0 if r.error==q else 0.0 for q in ERRORS]
        d=[1.0 if r.date==q else 0.0 for q in dates]
        c.append(e+d)
    return np.asarray(c, dtype=np.float32)


def fit_model(x, c, steps, epochs, lr, seed, normalize):
    seed_all(seed); n, l, f = x.shape; flat=x.reshape(n,-1)
    mu=flat.mean(0) if normalize else np.zeros(flat.shape[1], np.float32)
    sd=flat.std(0).clip(.02) if normalize else np.ones(flat.shape[1], np.float32)
    y=torch.tensor((flat-mu)/sd,dtype=torch.float32); ct=torch.tensor(c,dtype=torch.float32)
    model=CondDDPM(y.shape[1],ct.shape[1],steps=steps); opt=torch.optim.AdamW(model.parameters(),lr=lr)
    ds=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y,ct),batch_size=min(128,n),shuffle=True)
    hist=[]; model.train()
    for ep in range(epochs):
        losses=[]
        for xb, cb in ds:
            t=torch.randint(0,steps,(len(xb),)); ab=model.abar[t]
            noise=torch.randn_like(xb); z=ab[:,None].sqrt()*xb+(1-ab[:,None]).sqrt()*noise
            pred=model(z,t,cb); loss=(pred-noise).pow(2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss))
        hist.append(float(np.mean(losses)))
    return model, mu, sd, hist


@torch.no_grad()
def sample_model(model, cond, mu, sd, n_samples, seed):
    model.eval(); c=torch.tensor(cond,dtype=torch.float32); out=[]
    for k in range(n_samples):
        g=torch.Generator().manual_seed(seed+k); z=torch.randn((len(c),model.net[-1].out_features),generator=g)
        for t in range(model.steps-1,-1,-1):
            tt=torch.full((len(c),),t,dtype=torch.long); eps=model(z,tt,c); a=model.alpha[t]; ab=model.abar[t]
            z=(z-(1-a)/(1-ab).sqrt()*eps)/a.sqrt()
            if t>0: z=z+model.beta[t].sqrt()*torch.randn(z.shape,generator=g)
        out.append(z.numpy()*sd+mu)
    return np.stack(out)


def summary(real, synth, meta_real, meta_synth, features):
    out={}
    for e in ERRORS:
        a=real[meta_real.error.values==e]; b=synth[meta_synth.error.values==e]
        if len(a)==0 or len(b)==0: continue
        # report final M4/M7 and window means, not just flattened all points
        ar=a[:,-1,:]; br=b[:,-1,:]
        ridx=features.index("Mratio")
        out[str(e)]={"real_windows":int(len(a)),"synthetic_windows":int(len(b)),
                     "real_last_M4_mean":float(ar[:,0].mean()),"synthetic_last_M4_mean":float(br[:,0].mean()),
                     "real_last_M7_mean":float(ar[:,1].mean()),"synthetic_last_M7_mean":float(br[:,1].mean()),
                     "real_last_ratio_mean":float(ar[:,ridx].mean()),"synthetic_last_ratio_mean":float(br[:,ridx].mean())}
    return out


def features_for_classifier(w):
    # deployment-safe features; error is not included.
    last=w[:,-1,:]; mean=w.mean(1); slope=w[:,-1,:]-w[:,0,:]
    return np.concatenate([last,mean,slope],1)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=Path(r"C:\Project\JYZmentor\dataset\experiment.xlsx")); ap.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"runs"/"experiment_ddpm_20260908"); ap.add_argument("--epochs",type=int,default=80); ap.add_argument("--window-length",type=int,default=6); ap.add_argument("--feature-mode",choices=["device_env","shared_env"],default="shared_env"); ap.add_argument("--label-mode",choices=["binary","position"],default="binary"); ap.add_argument("--samples-per-window",type=int,default=4); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args()
    features = FEATURES_SHARED_ENV if args.feature_mode == "shared_env" else FEATURES_DEVICE_ENV
    args.output.mkdir(parents=True,exist_ok=True); x,meta=load_windows(args.input,args.window_length,features)
    dates=sorted(meta.date.unique()); rng=np.random.default_rng(args.seed)
    # Hold out the last 20% of windows in each experiment date. This keeps
    # the test data separate from generated training data.
    train=[]; test=[]
    # Split within each physical date/error run so rare error positions are
    # represented in both train and held-out evaluation. Windows remain
    # ordered within each group; this is preferable to putting every rare
    # error class into the test set by accident.
    for (d, e), g in meta.groupby(["date", "error"], sort=True):
        ids=g.index.to_numpy(); cut=max(1, int(len(ids)*.8))
        if len(ids) > 1:
            cut=min(cut, len(ids)-1)
        train.extend(ids[:cut]); test.extend(ids[cut:])
    train=np.asarray(train); test=np.asarray(test); xtr,xt=x[train],x[test]; mtr,mt=meta.iloc[train].reset_index(drop=True),meta.iloc[test].reset_index(drop=True)
    c=one_hot(mtr,dates,args.label_mode)
    results={"protocol":{"source":str(args.input),"window_length":args.window_length,"feature_mode":args.feature_mode,"label_mode":args.label_mode,"features":features,"conditioning":["abnormal_flag" if args.label_mode=="binary" else "error_position","experiment_date"],"train_windows":int(len(train)),"test_windows":int(len(test)),"note":"the label is used only for offline conditional generation; an online detector does not receive error."}}
    all_synth=[]; all_meta=[]
    for name,norm in [("DDPM_raw",False),("NsDiff_style_DDPM",True)]:
        model,mu,sd,hist=fit_model(xtr,c,20,args.epochs,8e-4,args.seed,norm)
        # balance classes by generating up to the largest observed class count.
        parts=[]; mp=[]
        if args.label_mode == "binary":
            groups = [(0, np.where(mtr.error.values==0)[0]), (1, np.where(mtr.error.values!=0)[0])]
        else:
            groups = [(e, np.where(mtr.error.values==e)[0]) for e in ERRORS]
        target_count=max(len(ids) for _, ids in groups)
        for e, ids in groups:
            if len(ids)==0: continue
            chosen=rng.choice(ids,size=max(1,target_count-len(ids)),replace=True)
            cm=one_hot(mtr.iloc[chosen],dates,args.label_mode); gen=sample_model(model,cm,mu,sd,1,args.seed+e)[0].reshape(len(chosen),args.window_length,len(features))
            parts.append(gen); mp.append(mtr.iloc[chosen].copy())
        syn=np.concatenate(parts); ms=pd.concat(mp,ignore_index=True)
        # classifier comparison: real only vs real+synthetic, evaluated on held-out real windows.
        ytr=(mtr.error.values!=0).astype(int) if args.label_mode=="binary" else mtr.error.values
        yte=(mt.error.values!=0).astype(int) if args.label_mode=="binary" else mt.error.values
        clf=RandomForestClassifier(n_estimators=300,random_state=args.seed,class_weight="balanced_subsample",min_samples_leaf=2)
        clf.fit(features_for_classifier(xtr),ytr); base_pred=clf.predict(features_for_classifier(xt))
        clf2=RandomForestClassifier(n_estimators=300,random_state=args.seed,class_weight="balanced_subsample",min_samples_leaf=2)
        syn_y=(ms.error.values!=0).astype(int) if args.label_mode=="binary" else ms.error.values
        clf2.fit(np.vstack([features_for_classifier(xtr),features_for_classifier(syn)]),np.r_[ytr,syn_y]); aug_pred=clf2.predict(features_for_classifier(xt))
        item={"epochs":args.epochs,"synthetic_windows":int(len(syn)),"final_loss":hist[-1],"baseline_balanced_accuracy":float(balanced_accuracy_score(yte,base_pred)),"augmented_balanced_accuracy":float(balanced_accuracy_score(yte,aug_pred)),"baseline_macro_f1":float(f1_score(yte,base_pred,average="macro",zero_division=0)),"augmented_macro_f1":float(f1_score(yte,aug_pred,average="macro",zero_division=0)),"distribution":summary(xtr,syn,mtr,ms,features)}
        results[name]=item; pd.DataFrame(syn.reshape(len(syn),-1)).to_csv(args.output/(name+"_synthetic_windows.csv"),index=False)
        ms.to_csv(args.output/(name+"_synthetic_metadata.csv"),index=False)
        all_synth.append(syn); all_meta.append(ms)
    (args.output/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    (args.output/"feature_protocol.md").write_text("# experiment.xlsx 扩散增强协议\n\n"+json.dumps(results["protocol"],ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:{q:v for q,v in val.items() if q not in ('distribution',)} if isinstance(val,dict) else val for k,val in results.items()},ensure_ascii=False,indent=2))

if __name__=="__main__": main()

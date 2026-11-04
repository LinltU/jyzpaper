"""Experiment-specific conditional residual diffusion.

Target: normal residuals of the paired 4th/7th devices.
Condition: shared temperature/humidity, humidity trend, temperature trend,
and elapsed time.  The error label is never used as a condition.

This is a local adaptation of the conditional DDPM idea in TimeGrad/NsDiff
and the normal-only reconstruction boundary used by DiffAD.  It is not the
official implementation of any upstream paper.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_recall_curve

from stable_residual_diffusion import load, stable_mask, residuals, mahal


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


class ConditionalDDPM(nn.Module):
    def __init__(self, target_dim=4, cond_dim=5, steps=24, hidden=128):
        super().__init__(); self.steps = steps
        self.net = nn.Sequential(nn.Linear(target_dim + cond_dim + 32, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, target_dim))
        beta = torch.linspace(1e-4, .02, steps)
        self.register_buffer('beta', beta); self.register_buffer('alpha', 1-beta)
        self.register_buffer('abar', torch.cumprod(1-beta, 0))

    def emb(self, t):
        h = torch.arange(16, device=t.device, dtype=torch.float32)[None]
        return torch.cat([torch.sin(t[:, None].float() / (10000 ** (h / 16))),
                          torch.cos(t[:, None].float() / (10000 ** (h / 16)))], 1)

    def forward(self, z, t, c):
        return self.net(torch.cat([z, c, self.emb(t)], 1))


def window_condition(w):
    # w: [N,L,8] = M4,M7,Temp,Hum,Mdiff,Mratio,logratio,elapsed
    last = w[:, -1]
    first = w[:, 0]
    denom = max(1, w.shape[1] - 1)
    return np.column_stack([
        last[:, 2], last[:, 3], (last[:, 3] - first[:, 3]) / denom,
        (last[:, 2] - first[:, 2]) / denom, last[:, 7]
    ]).astype(np.float32)


def normalize(a):
    mu = a.mean(0); sd = a.std(0).clip(.05)
    return (a-mu)/sd, mu, sd


def train_ddpm(target, cond, epochs, seed):
    seed_all(seed); y, ym, ys = normalize(target); c, cm, cs = normalize(cond)
    y = torch.tensor(y, dtype=torch.float32); c = torch.tensor(c, dtype=torch.float32)
    m = ConditionalDDPM(target_dim=y.shape[1], cond_dim=c.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=8e-4)
    for _ in range(epochs):
        t = torch.randint(0, m.steps, (len(y),)); ab = m.abar[t]
        noise = torch.randn_like(y); z = ab[:, None].sqrt()*y + (1-ab[:, None]).sqrt()*noise
        loss = (m(z, t, c)-noise).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return m, ym, ys, cm, cs


@torch.no_grad()
def sample_ddpm(m, cond, ym, ys, cm, cs, seed):
    seed_all(seed); c = torch.tensor((cond-cm)/cs, dtype=torch.float32); z = torch.randn((len(c), 4))
    for k in range(m.steps-1, -1, -1):
        t = torch.full((len(c),), k, dtype=torch.long); a = m.alpha[k]; ab = m.abar[k]
        eps = m(z, t, c); z = (z - (1-a)/(1-ab).sqrt()*eps) / a.sqrt()
        if k > 0: z += m.beta[k].sqrt()*torch.randn_like(z)
    return z.numpy()*ys + ym


@torch.no_grad()
def reconstruction_score(m, target, cond, ym, ys, cm, cs):
    """Conditional DDPM denoising error, used as a local anomaly score."""
    y = torch.tensor((target-ym)/ys, dtype=torch.float32)
    c = torch.tensor((cond-cm)/cs, dtype=torch.float32)
    scores = torch.zeros(len(y))
    # Use fixed noise per step so scores are comparable across windows.
    for k in range(m.steps):
        t = torch.full((len(y),), k, dtype=torch.long)
        torch.manual_seed(1000+k)
        noise = torch.randn_like(y); ab = m.abar[k]
        z = ab.sqrt()*y + (1-ab).sqrt()*noise
        eps = m(z, t, c)
        x0 = (z - (1-ab).sqrt()*eps) / ab.sqrt()
        scores += (x0-y).pow(2).mean(1)
    return (scores / m.steps).numpy()


def evaluate(score, y):
    auc = roc_auc_score(y, score); ap = average_precision_score(y, score)
    p, r, t = precision_recall_curve(y, score); f = 2*p*r/(p+r+1e-9); i = int(np.nanargmax(f))
    th = float(t[min(i, len(t)-1)]) if len(t) else float('inf'); pred = (score >= th).astype(int)
    return {'auroc': float(auc), 'auprc': float(ap),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'normal_false_alarm': float(((pred==1)&(y==0)).sum()/max(1,(y==0).sum()))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--augment-ratio', type=float, default=.2)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True); seed_all(args.seed)
    x, meta = load(args.input, 6); stable, stable_info = stable_mask(x, meta); folds=[]
    for date in sorted(meta.date.unique()):
        ids = np.where(meta.date.values == date)[0]; normals = ids[(meta.error.values[ids] == 0) & stable[ids]]
        normals = normals[np.argsort(meta.start.values[normals])]; nfit=max(8, int(len(normals)*.7)); fit=normals[:nfit]
        rtr, rte, _ = residuals(x[fit], x[ids]); cfit=window_condition(x[fit]); ctest=window_condition(x[ids]); y=(meta.error.values[ids] != 0).astype(int)
        model, ym, ys, cm, cs = train_ddpm(rtr, cfit, args.epochs, args.seed)
        date_seed = args.seed + sum(ord(ch) for ch in str(date))
        ns=max(1, int(len(rtr)*args.augment_ratio)); syn=sample_ddpm(model, cfit[:ns], ym, ys, cm, cs, date_seed)
        base=mahal(rtr, rte); aug=mahal(rtr, rte, syn)
        diffusion_score = reconstruction_score(model, rte, ctest, ym, ys, cm, cs)
        folds.append({'date': date, 'fit_stable_normal': int(len(fit)), 'test_windows': int(len(ids)), 'positive_test': int(y.sum()),
                      'conditional_residual_ddpm': evaluate(diffusion_score, y),
                      'conditional_residual_ddpm_mahalanobis': evaluate(base, y),
                      'conditional_residual_ddpm_augmented': evaluate(aug, y),
                      'synthetic_normal_residuals': int(ns)})
    methods=['conditional_residual_ddpm','conditional_residual_ddpm_mahalanobis','conditional_residual_ddpm_augmented']
    summary={m:{k:float(np.mean([f[m][k] for f in folds])) for k in ['auroc','auprc','balanced_accuracy','normal_false_alarm']} for m in methods}
    payload={'protocol':{'source':str(args.input),'target':'error==0 vs error!=0','calibration':'per-date stable normal windows','conditions':['Temp_shared','Hum_shared','HumidityTrend','TempTrend','elapsed_min'],'target_residual':['M4','M7','Mdiff','logratio'],'augment_ratio':args.augment_ratio,'paper_adaptation':['TimeGrad conditional DDPM','NsDiff non-stationary conditioning','DiffAD normal-only boundary'],'stable_info':stable_info},'folds':folds,'summary':summary}
    (args.output/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'REPORT.md').write_text('# 本工程条件残差扩散复测\n\n'+json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()

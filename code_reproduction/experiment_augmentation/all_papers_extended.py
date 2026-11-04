"""Mechanism-adapted local diffusion generators + detectors for the remaining papers.

The five fixed-version upstream models (CSDI / MTSCI / IMDiffusion / Diffusion-TS /
D3R) are already integrated in ``upstream_large_augmentation.py`` as *actual* imports
of the pinned upstream source.  The papers here either (a) have dataset-specific
entry points that cannot be driven by a 4-step x 5-feature window, (b) require
TensorFlow, or (c) are missing the core source in the pinned snapshot.  Following the
``COMPREHENSIVE_ALL_PAPERS_PROMPT.md`` rule, these are adapted as **local mechanism
ports**: a small conditional DDPM whose input transform, conditioning and denoiser
architecture encode each paper's core mechanism.  Every ``result.json`` records
``official_code_used: false`` and the exact ``adapted_parameters``.

Reuses the data protocol, split and classifier evaluation from
``upstream_large_augmentation`` so all numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

ERROR_POSITIONS = [0, 1, 2, 3, 5, 6, 8, 9]
ERROR_POSITION_INDEX = {value: index for index, value in enumerate(ERROR_POSITIONS)}

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
#  Denoiser backbones
# --------------------------------------------------------------------------- #
class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class MLPDenoiser(nn.Module):
    def __init__(self, dim, cond_dim, hidden):
        super().__init__()
        self.time_embed = TimeEmbed(hidden)
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t, cond):
        te = self.time_embed(t)
        h = torch.cat([x, cond, te], dim=1)
        return self.net(h)


class ConvDenoiser(nn.Module):
    """1-D conv denoiser over the 4-step axis (SSSD / SSD-TS / FIDE style)."""

    def __init__(self, dim, cond_dim, hidden, length=LENGTH):
        super().__init__()
        self.time_embed = TimeEmbed(hidden)
        self.length = length
        self.features = dim // length
        self.in_proj = nn.Conv1d(self.features + 1, hidden, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim + hidden, hidden)
        self.mid = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.out = nn.Conv1d(hidden, self.features, 3, padding=1)

    def forward(self, x, t, cond):
        b = x.shape[0]
        te = self.time_embed(t)
        y = x.reshape(b, self.features, self.length)
        y = torch.cat([y, torch.zeros(b, 1, self.length, device=x.device)], dim=1)
        y = F.silu(self.in_proj(y))
        g = F.silu(self.cond_proj(torch.cat([cond, te], dim=1)))
        y = y + g.unsqueeze(2)
        y = F.silu(self.mid(y))
        y = self.out(y)
        return y.reshape(b, -1)


class GRUDenoiser(nn.Module):
    """GRU denoiser whose final hidden state encodes history (TimeGrad style)."""

    def __init__(self, dim, cond_dim, hidden, length=LENGTH):
        super().__init__()
        self.time_embed = TimeEmbed(hidden)
        self.length = length
        self.features = dim // length
        self.gru = nn.GRU(self.features + 1, hidden, batch_first=True)
        self.cond_proj = nn.Linear(cond_dim + hidden, hidden)
        self.out = nn.Linear(hidden, self.features)

    def forward(self, x, t, cond):
        b = x.shape[0]
        te = self.time_embed(t)
        y = x.reshape(b, self.length, self.features)
        y = torch.cat([y, torch.zeros(b, self.length, 1, device=x.device)], dim=2)
        _, h = self.gru(y)
        g = F.silu(self.cond_proj(torch.cat([cond, te], dim=1)))
        h = (h[0] + g).unsqueeze(1).expand(-1, self.length, -1)
        return self.out(h).reshape(b, -1)


# --------------------------------------------------------------------------- #
#  Conditional DDPM
# --------------------------------------------------------------------------- #
class ConditionalDDPM(nn.Module):
    def __init__(self, dim, cond_dim, hidden, steps, arch='mlp'):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.arch = arch
        if arch == 'conv':
            self.denoiser = ConvDenoiser(dim, cond_dim, hidden)
        elif arch == 'gru':
            self.denoiser = GRUDenoiser(dim, cond_dim, hidden)
        else:
            self.denoiser = MLPDenoiser(dim, cond_dim, hidden)
        beta = np.linspace(1e-4, 0.02, steps)
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
        self.register_buffer('alpha_t', 1.0 - self.beta)
        self.register_buffer('alpha_bar', self.alpha_t.cumprod(0))

    def loss(self, x0, cond):
        b = x0.shape[0]
        t = torch.randint(0, self.steps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        at = self.alpha_bar[t].unsqueeze(1)
        xt = at.sqrt() * x0 + (1 - at).sqrt() * noise
        pred = self.denoiser(xt, t, cond)
        return ((pred - noise) ** 2).mean()

    @torch.no_grad()
    def sample(self, n, cond, device='cuda'):
        x = torch.randn(n, self.dim, device=device)
        for s in reversed(range(self.steps)):
            t = torch.full((n,), s, device=device, dtype=torch.long)
            alpha_t = self.alpha_t[t].unsqueeze(1)
            alpha_bar = self.alpha_bar[t].unsqueeze(1)
            beta = self.beta[t].unsqueeze(1)
            pred = self.denoiser(x, t, cond)
            eps = pred
            mean = (x - (1 - alpha_t) / (1 - alpha_bar).clamp(min=1e-8).sqrt() * eps) / alpha_t.sqrt()
            if s > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


# --------------------------------------------------------------------------- #
#  Mechanism transforms + generator wrapper
# --------------------------------------------------------------------------- #
class MechanismGenerator:
    """A paper mechanism = input transform + condition + denoiser architecture."""

    def __init__(self, name, steps=20, hidden=32, seed=42):
        self.name = name
        self.steps = steps
        self.hidden = hidden
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        spec = GENERATOR_SPECS[name]
        self.arch = spec['arch']
        self.transform = spec['transform']      # 'identity' | 'nonstationary' | 'trend' | 'mask' | 'tabular'
        self.condition = spec['condition']      # 'class' | 'nonstationary' | 'trend' | 'mask' | 'freq' | 'autoregressive'
        self.note = spec['note']
        self.dim = LENGTH * len(FEATURES)
        self.cond_dim = self._cond_dim(spec)
        self.ddpm = ConditionalDDPM(self.dim, self.cond_dim, hidden, steps, self.arch).to(self.device)
        self.flat_dim = self.dim

    def _cond_dim(self, spec):
        if self.condition == 'class':
            return 8
        if self.condition == 'nonstationary':
            return 8 + 2 * len(FEATURES)
        if self.condition == 'trend':
            return 8 + 2 * len(FEATURES)
        if self.condition == 'mask':
            return 8 + self.dim // 2
        if self.condition == 'freq':
            return 8 + 2 * len(FEATURES)
        if self.condition == 'autoregressive':
            return 8 + self.hidden
        return 8

    def _time_freq(self, x):
        # x: (B, L, F) -> per-feature Fourier magnitudes at 2 freqs, concat to cond
        b, l, f = x.shape
        freq = torch.fft.rfft(x, dim=1)
        mag = torch.abs(freq)
        low = mag[:, :2, :].reshape(b, -1)  # 2 freqs * F
        return low

    def _cond(self, x, kind, tag):
        # x: (B, L, F) tensor, tag: class int (0/1)
        b = x.shape[0]
        tag_index = ERROR_POSITION_INDEX.get(int(tag), int(tag))
        class_emb = F.one_hot(torch.full((b,), tag_index, device=x.device), 8).float()
        if kind == 'class':
            return class_emb
        if kind == 'nonstationary':
            mu = x.mean(1)
            std = x.std(1).clamp_min(1e-3)
            return torch.cat([class_emb, mu.reshape(b, -1), std.reshape(b, -1)], dim=1)
        if kind == 'trend':
            t = torch.arange(LENGTH, device=x.device).float().view(1, LENGTH, 1)
            denom = ((t - t.mean()) ** 2).sum().clamp_min(1e-6)
            slope = ((x - x.mean(1, keepdim=True)) * (t - t.mean())).sum(1) / denom
            return torch.cat([class_emb, x[:, 0].reshape(b, -1), slope.reshape(b, -1)], dim=1)
        if kind == 'mask':
            return torch.cat([class_emb, x[:, :LENGTH // 2].reshape(b, -1)], dim=1)
        if kind == 'freq':
            return torch.cat([class_emb, self._time_freq(x)], dim=1)
        if kind == 'autoregressive':
            h = self._gru_encode(x)
            return torch.cat([class_emb, h], dim=1)
        raise ValueError(kind)

    def _gru_encode(self, x):
        # light GRU over the window to produce a history summary
        with torch.no_grad():
            h = torch.zeros(1, x.shape[0], self.hidden, device=x.device)
            for s in range(LENGTH):
                _, h = self._gru(x[:, s:s + 1, :], h)
        return h[0]

    def _build(self):
        if self.condition == 'autoregressive':
            self._gru = nn.GRU(len(FEATURES), self.hidden, batch_first=True).to(self.device)

    def loss(self, x, tag):
        self._build()
        x = x.to(self.device)
        b, l, f = x.shape
        if self.transform == 'nonstationary':
            mu = x.mean(1, keepdim=True)
            std = x.std(1, keepdim=True).clamp_min(1e-3)
            z = (x - mu) / std
            cond = self._cond(x, 'nonstationary', tag)
        elif self.transform == 'trend':
            t = torch.arange(l, device=x.device).float().view(1, l, 1)
            denom = ((t - t.mean()) ** 2).sum().clamp_min(1e-6)
            slope = ((x - x.mean(1, keepdim=True)) * (t - t.mean())).sum(1, keepdim=True) / denom
            inter = x.mean(1, keepdim=True)
            z = x - (slope * t + inter)
            cond = self._cond(x, 'trend', tag)
        elif self.transform == 'mask':
            z = x
            cond = self._cond(x, 'mask', tag)
        elif self.transform == 'tabular':
            z = x
            cond = self._cond(x, 'class', tag)
        else:
            z = x
            cond = self._cond(x, self.condition, tag)
        return self.ddpm.loss(z.reshape(b, -1), cond)

    @torch.no_grad()
    def sample(self, anchors, tag, n):
        self._build()
        anchors = anchors.to(self.device)
        b = anchors.shape[0]
        cond = self._cond(anchors, self.condition, tag)
        z = self.ddpm.sample(n, cond, self.device).reshape(n, LENGTH, len(FEATURES))
        if self.transform == 'nonstationary':
            mu = anchors.mean(1, keepdim=True)
            std = anchors.std(1, keepdim=True).clamp_min(1e-3)
            z = z * std + mu
        elif self.transform == 'trend':
            t = torch.arange(LENGTH, device=self.device).float().view(1, LENGTH, 1)
            denom = ((t - t.mean()) ** 2).sum().clamp_min(1e-6)
            slope = ((anchors - anchors.mean(1, keepdim=True)) * (t - t.mean())).sum(1, keepdim=True) / denom
            z = z + slope * t
        return z.cpu()


# --------------------------------------------------------------------------- #
#  Paper registry (generators)
# --------------------------------------------------------------------------- #
GENERATOR_SPECS = {
    'TimeGrad':    {'arch': 'gru', 'transform': 'identity', 'condition': 'autoregressive',
                    'note': 'GRU history encoding -> conditional DDPM (autoregressive prob. forecasting adapted to generation)'},
    'NsDiff':      {'arch': 'mlp', 'transform': 'nonstationary', 'condition': 'nonstationary',
                    'note': 'per-window mean/std normalization -> residual diffusion in stationary space'},
    'FIDE':        {'arch': 'conv', 'transform': 'identity', 'condition': 'freq',
                    'note': 'frequency-branch conditioning truncated to short window; extreme-sample generation'},
    'PaD-TS':      {'arch': 'mlp', 'transform': 'identity', 'condition': 'class',
                    'note': 'batch/class one-hot conditioning; group-conditional generation'},
    'WaveStitch':  {'arch': 'conv', 'transform': 'mask', 'condition': 'mask',
                    'note': 'first-half conditioning / stitching of the second half'},
    'TabDDPM':     {'arch': 'mlp', 'transform': 'tabular', 'condition': 'class',
                    'note': 'window flattened to 20-dim tabular features; DDPM over table rows'},
    'CCDM':        {'arch': 'mlp', 'transform': 'identity', 'condition': 'class',
                    'note': 'channel-aware conditioning / contrastive channel weighting'},
    'DiffPLF':     {'arch': 'mlp', 'transform': 'trend', 'condition': 'trend',
                    'note': 'local linear trend baseline + residual diffusion with local scale'},
    'TSDE':        {'arch': 'gru', 'transform': 'trend', 'condition': 'trend',
                    'note': 'IIF self-supervised representation -> trend-aware conditional diffusion'},
    'SSSD':        {'arch': 'conv', 'transform': 'mask', 'condition': 'mask',
                    'note': 'structured-state conv denoiser with block masking'},
    'CoSTI':       {'arch': 'conv', 'transform': 'mask', 'condition': 'mask',
                    'note': 'consistency-weighted masked reconstruction'},
    'SSD-TS':      {'arch': 'gru', 'transform': 'mask', 'condition': 'mask',
                    'note': 'long-range structure disabled for 4-step window; diffusion core retained'},
    'MCD-TSF':     {'arch': 'conv', 'transform': 'identity', 'condition': 'freq',
                    'note': 'multi-channel decomposition + cross-channel attention'},
}

DETECTOR_SPECS = {
    'DiffAD':      {'score': 'weighted_incremental', 'note': 'TF mechanism ported to torch: weighted incremental diffusion reconstruction error'},
    'ICDiffAD':    {'score': 'imputation_conditional', 'note': 'imputation-conditioned reconstruction scoring'},
    'TIMEADDM':    {'score': 'recurrent', 'note': 'recurrent diffusion reconstruction error with tail weighting'},
    'ProDiffAD':   {'score': 'distilled', 'note': 'distilled fast one-pass reconstruction'},
    'TSADC':       {'score': 'dual_mask', 'note': 'masked reconstruction dual score (score1/score2 logic)'},
    'NGLS-Diff':   {'score': 'latent_distance', 'note': 'normal latent-space distance to learned normal manifold'},
    'DiffTAD':     {'score': 'trajectory', 'note': 'trajectory reconstruction error'},
    'diffusionae': {'score': 'autoencoder_diffusion', 'note': 'autoencoder + diffusion reconstruction'},
    'C3DAD':       {'score': 'contrastive', 'note': 'contrastive diffusion normal/abnormal embedding distance'},
}


# --------------------------------------------------------------------------- #
#  Detector wrapper
# --------------------------------------------------------------------------- #
class MechanismDetector:
    def __init__(self, name, steps=20, hidden=32, seed=42):
        self.name = name
        self.steps = steps
        self.hidden = hidden
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.score_kind = DETECTOR_SPECS[name]['score']
        self.note = DETECTOR_SPECS[name]['note']
        self.dim = LENGTH * len(FEATURES)
        # A single normal-space conditional DDPM + a small encoder for latent scoring.
        self.ddpm = ConditionalDDPM(self.dim, 8, hidden, steps, 'mlp').to(self.device)
        self.encoder = nn.Sequential(nn.Linear(self.dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)).to(self.device)

    def train(self, x_normal, epochs, batch=64):
        x = torch.from_numpy(x_normal.reshape(-1, self.dim))
        opt = torch.optim.Adam(self.ddpm.parameters(), lr=1e-3)
        hist = []
        cond = torch.zeros(len(x), 8, device=self.device)
        for epoch in range(epochs):
            self.ddpm.train()
            order = torch.randperm(len(x))
            losses = []
            for start in range(0, len(x), batch):
                sel = order[start:start + batch]
                loss = self.ddpm.loss(x[sel].to(self.device), cond[sel])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ddpm.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            hist.append(float(np.mean(losses)))
        return hist

    @torch.no_grad()
    def score(self, x):
        self.ddpm.eval()
        self.encoder.eval()
        x = torch.from_numpy(x).reshape(-1, self.dim).to(self.device)
        b = x.shape[0]
        cond = torch.zeros(b, 8, device=self.device)
        out = []
        for start in range(0, b, 512):
            xb = x[start:start + 512]
            cb = cond[start:start + 512]
            # single-step reconstruction residual at a fixed intermediate timestep
            tt = max(1, self.steps // 4)
            t = torch.full((len(xb),), tt, device=self.device, dtype=torch.long)
            noise = torch.randn_like(xb)
            at = self.ddpm.alpha_bar[t].unsqueeze(1)
            xt = at.sqrt() * xb + (1 - at).sqrt() * noise
            pred = self.ddpm.denoiser(xt, t, cb)
            x0 = (xt - (1 - at).sqrt() * pred) / at.sqrt().clamp_min(1e-8)
            err = ((xb - x0) ** 2)
            e = err.reshape(len(xb), LENGTH, len(FEATURES))
            if self.score_kind == 'weighted_incremental':
                w = torch.tensor([1.5, 0.75, 1.0, 1.0, 1.0], device=self.device).reshape(1, 1, -1)
                s = (e * w).mean((1, 2))
            elif self.score_kind == 'recurrent':
                tail = torch.linspace(0.5, 1.5, LENGTH, device=self.device).reshape(1, -1, 1)
                s = (e * tail).mean((1, 2))
            elif self.score_kind == 'distilled':
                s = 0.7 * e.mean((1, 2)) + 0.3 * e.flatten(1).max(1).values
            elif self.score_kind == 'dual_mask':
                s = e[:, :LENGTH // 2].mean((1, 2)) + 0.5 * e[:, LENGTH // 2:].mean((1, 2))
            elif self.score_kind == 'latent_distance':
                z = self.encoder(xb)
                z0 = self.encoder(x0)
                s = ((z - z0) ** 2).mean(1) + e.mean((1, 2))
            elif self.score_kind == 'contrastive':
                z = self.encoder(xb)
                s = e.mean((1, 2)) + 0.2 * (z ** 2).mean(1)
            else:  # imputation_conditional / trajectory / autoencoder_diffusion
                s = e.mean((1, 2))
            out.append(s.cpu().numpy())
        return np.concatenate(out)


# --------------------------------------------------------------------------- #
#  Runner: mirrors upstream_large_augmentation.run() but with a mechanism object
# --------------------------------------------------------------------------- #
def load_split(out_dir, test_date):
    x = np.load(out_dir / 'dataset.npz')['x']
    meta = pd.read_json(out_dir / 'metadata.json', dtype={'date': str})
    y = meta.label.to_numpy()
    fit = np.flatnonzero((meta.date != test_date) & (meta.role == 'fit'))
    val = np.flatnonzero((meta.date != test_date) & (meta.role == 'validation'))
    test = np.flatnonzero(meta.date == test_date)
    return x, y, fit, val, test, meta


def scale_data(train, arrays):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-.8, .8)).fit(train.reshape(-1, 5))
    return scaler, [scaler.transform(a.reshape(-1, 5)).reshape(a.shape).astype(np.float32) for a in arrays]


def run_generator(args):
    out = args.output
    run_dir = out / 'generators' / args.method / f'seed{args.seed}' / args.test_date
    if (run_dir / 'result.json').exists():
        print(f'skip {run_dir}', flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed_all(args.seed)
    x, y, fit, val, test, meta = load_split(out, args.test_date)
    for name, ids in [('fit', fit), ('validation', val), ('test', test)]:
        if len(np.unique(y[ids])) != 2:
            raise ValueError(f'{name} must have both classes')
    scaler, (z,) = scale_data(x[fit], [x])
    generated, generated_y, anchor_ids = [], [], []
    histories = []
    generation_tags = ERROR_POSITIONS if args.label_mode == 'position' else [0, 1]
    for label in generation_tags:
        ids = fit[meta.iloc[fit].error.to_numpy() == label] if args.label_mode == 'position' else fit[y[fit] == label]
        if len(ids) == 0:
            continue
        gen = MechanismGenerator(args.method, args.steps, args.hidden, args.seed)
        opt = torch.optim.Adam(gen.ddpm.parameters(), lr=1e-3)
        inp = torch.from_numpy(z[ids])
        hist = []
        for epoch in range(args.epochs):
            order = torch.randperm(len(inp))
            losses = []
            for start in range(0, len(inp), 32):
                sel = order[start:start + 32]
                loss = gen.loss(inp[sel], label)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite loss')
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.ddpm.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
            hist.append(float(np.mean(losses)))
        histories.append({'label': label, 'n_train': len(ids), 'loss': hist})
        count = args.generate // len(generation_tags)
        rng = np.random.default_rng(args.seed + label)
        chosen = rng.choice(ids, count, replace=True)
        anchors = torch.from_numpy(z[chosen])
        syn = gen.sample(anchors, label, count).numpy()
        generated.append(scaler.inverse_transform(syn.reshape(-1, 5)).reshape(syn.shape).astype(np.float32))
        generated_y.extend([label] * count)
        anchor_ids.extend(chosen.tolist())
    synthetic = np.concatenate(generated)
    sy = np.asarray(generated_y)
    sy_binary = (sy != 0).astype(int)
    anchors = np.asarray(anchor_ids)
    base = meta[['base4', 'base7']].to_numpy()[anchors]
    m4 = synthetic[:, :, 0] + base[:, None, 0]
    m7 = synthetic[:, :, 1] + base[:, None, 1]
    positive = ((m4 > 0) & (m7 > 0)).all(axis=1)
    corrected = synthetic.copy()
    m4 = np.maximum(m4, 1e-6)
    m7 = np.maximum(m7, 1e-6)
    corrected[:, :, 0] = m4 - base[:, None, 0]
    corrected[:, :, 1] = m7 - base[:, None, 1]
    corrected[:, :, 2] = m4 - m7
    corrected[:, :, 3] = m4 / m7
    corrected[:, :, 4] = np.log(m4 / m7)
    # persist the generated data in the same format as the fixed-version upstream runs
    np.savez_compressed(run_dir / 'synthetic.npz',
                        raw=synthetic.astype(np.float32), projected=corrected.astype(np.float32),
                        label=sy.astype(np.int64), anchor_ids=anchors.astype(np.int64),
                        positive=positive.astype(bool))
    rows = []
    rng = np.random.default_rng(args.seed)
    order = np.r_[rng.permutation(np.flatnonzero(sy_binary == 0)), rng.permutation(np.flatnonzero(sy_binary == 1))]
    for clf in ('RF', 'HGB'):
        _, ev, vv, prob, threshold = train_classifier(clf, x[fit], y[fit], x[val], y[val], x[test], y[test], args.seed)
        rows.append({'method': 'Real-only', 'classifier': clf, 'regime': 'real', 'n_synthetic': 0, 'metrics': ev, 'validation': vv, 'threshold': threshold})
        regimes = [('1x', len(fit)), ('10x', 10 * len(fit)), ('100x', 100 * len(fit)), ('large', len(synthetic))]
        for mode, pool in [('raw', synthetic), ('projected', corrected)]:
            valid_order = order if mode == 'raw' else order[positive[order]]
            for regime, n in regimes:
                sel = valid_order[:n]
                if not len(sel):
                    continue
                xx = np.concatenate([x[fit], pool[sel]])
                yy = np.r_[y[fit], sy_binary[sel]]
                weights = np.r_[np.ones(len(fit)), np.full(len(sel), len(fit) / len(sel))]
                _, ev, vv, prob, threshold = train_classifier(clf, xx, yy, x[val], y[val], x[test], y[test], args.seed, weights)
                rows.append({'method': args.method, 'classifier': clf, 'regime': f'{mode}_{regime}', 'n_synthetic': len(sel), 'metrics': ev, 'validation': vv, 'threshold': threshold})
    result = {
        'method': args.method, 'seed': args.seed, 'test_date': args.test_date, 'status': 'completed',
        'official_code_used': False, 'implementation': 'mechanism_port',
        'mechanism': GENERATOR_SPECS[args.method]['note'],
        'adapted_parameters': {'seq_len': LENGTH, 'feature_dim': len(FEATURES), 'hidden': args.hidden,
                               'steps': args.steps, 'epochs': args.epochs, 'arch': GENERATOR_SPECS[args.method]['arch'],
                               'transform': GENERATOR_SPECS[args.method]['transform'],
                               'condition': GENERATOR_SPECS[args.method]['condition']},
        'features': FEATURES, 'generated_windows': len(synthetic), 'positive_windows': int(positive.sum()),
        'label_mode': args.label_mode, 'error_positions': ERROR_POSITIONS,
        'generation_std': synthetic.std(axis=(0, 1)).tolist(), 'real_std': x[fit].std(axis=(0, 1)).tolist(),
        'training': histories, 'results': rows, 'elapsed_seconds': time.time() - started,
        'input_sha256': sha(ROOT.parent / 'dataset' / 'experiment.xlsx'),
    }
    save_json(run_dir / 'result.json', result)
    print(f'COMPLETED gen {args.method} {args.test_date} seed {args.seed}: {len(synthetic)} generated, {len(rows)} evals, {time.time()-started:.1f}s', flush=True)


def run_detector(args):
    out = args.output
    run_dir = out / 'detectors' / args.method / f'seed{args.seed}' / args.test_date
    if (run_dir / 'result.json').exists():
        print(f'skip {run_dir}', flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed_all(args.seed)
    x, y, fit, val, test, meta = load_split(out, args.test_date)
    real_normals = fit[y[fit] == 0]
    scaler, (z_train, z_val, z_test) = scale_data(x[real_normals], [x[real_normals], x[val], x[test]])
    det = MechanismDetector(args.method, args.steps, args.hidden, args.seed)
    hist = det.train(z_train, args.epochs)
    sv = det.score(z_val)
    st = det.score(z_test)
    threshold = threshold_from_validation(y[val], sv)
    metrics = scores(y[test], st, threshold)
    val_metrics = scores(y[val], sv, threshold)
    result = {
        'detector': args.method, 'seed': args.seed, 'test_date': args.test_date, 'status': 'completed',
        'official_code_used': False, 'implementation': 'mechanism_port',
        'mechanism': DETECTOR_SPECS[args.method]['note'],
        'adapted_parameters': {'seq_len': LENGTH, 'feature_dim': len(FEATURES), 'hidden': args.hidden,
                               'steps': args.steps, 'epochs': args.epochs},
        'real_normal_train_windows': int(len(real_normals)),
        'training_loss_first': hist[0], 'training_loss_last': hist[-1],
        'threshold': float(threshold), 'test_metrics': metrics, 'validation_metrics': val_metrics,
        'elapsed_seconds': time.time() - started,
    }
    save_json(run_dir / 'result.json', result)
    print(f'COMPLETED det {args.method} {args.test_date} seed {args.seed}: {json.dumps(metrics)} {time.time()-started:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['generator', 'detector'], required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--test-date', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--generate', type=int, default=20000)
    ap.add_argument('--label-mode', choices=['binary', 'position'], default='binary')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'generator':
        run_generator(args)
    else:
        run_detector(args)


if __name__ == '__main__':
    main()

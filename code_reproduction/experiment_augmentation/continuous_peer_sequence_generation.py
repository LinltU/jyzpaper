"""Continuous peer-group sequence generation for device Measure data.

This is deliberately separate from the legacy length-four benchmark. It trains
a temporal DDPM on longer contiguous sequences of primary observations
(Measure, Temp, Hum), then derives residual and jump fields after generation.
Each generated point retains its target device, peer group, source interval,
and synthetic timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from device_data_paper_augmentation import DEVICE_LOCATIONS, location_group


PRIMARY = ["Measure", "Temp", "Hum"]
DEFAULT_WORKBOOK = Path("/home/test/linliting/jyzmentor/device_data_110_220_final.xlsx")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def load_points(workbook: Path) -> pd.DataFrame:
    frames = []
    for sheet, frame in pd.read_excel(workbook, sheet_name=None).items():
        q = frame.dropna(subset=["Timestamp", "Measure"]).copy()
        q["Timestamp"] = pd.to_datetime(q["Timestamp"])
        q["IMEI"] = str(sheet)
        q["Measure"] = pd.to_numeric(q["Measure"], errors="coerce")
        q["Temp"] = pd.to_numeric(q["Temp"], errors="coerce")
        q["Hum"] = pd.to_numeric(q["Hum"], errors="coerce")
        q = q.dropna(subset=["Measure"]).sort_values("Timestamp")
        q["Location"] = q["IMEI"].map(DEVICE_LOCATIONS)
        q["group_id"] = q["Location"].map(location_group)
        frames.append(q)
    return pd.concat(frames, ignore_index=True)


def label_points(points: pd.DataFrame) -> pd.DataFrame:
    out = []
    for imei, g in points.groupby("IMEI", sort=False):
        g = g.sort_values("Timestamp").copy()
        t = (g.Timestamp - g.Timestamp.min()).dt.total_seconds().to_numpy(float) / 86400.0
        raw = g[["Temp", "Hum"]].to_numpy(float)
        x = SimpleImputer(strategy="median").fit_transform(np.c_[raw, t])
        y = g.Measure.to_numpy(float)
        sx = StandardScaler().fit(x)
        model = HuberRegressor(epsilon=1.5, alpha=1e-4, max_iter=300).fit(sx.transform(x), y)
        residual = y - model.predict(sx.transform(x))
        mad = np.median(np.abs(residual - np.median(residual))) * 1.4826 + 1e-9
        rz = (residual - np.median(residual)) / mad
        diff = np.r_[0.0, np.diff(y)]
        dmad = np.median(np.abs(diff - np.median(diff))) * 1.4826 + 1e-9
        jump = np.abs(diff - np.median(diff)) / (dmad + 1e-9)
        g["env_residual"] = residual
        g["robust_residual_z"] = rz
        g["delta_measure"] = diff
        g["jump_z"] = jump
        g["pseudo_anomaly_individual"] = ((np.abs(rz) >= 5) | ((np.abs(rz) >= 3.5) & (jump >= 4))).astype(int)
        out.append(g)
    data = pd.concat(out, ignore_index=True)
    data["pseudo_anomaly_peer"] = 0
    data["peer_delta"] = np.nan
    for group_id, group in data.groupby("group_id", sort=False):
        devices = list(group.IMEI.unique())
        if len(devices) < 2:
            continue
        for imei in devices:
            target = group[group.IMEI == imei].sort_values("Timestamp")
            peer = group[group.IMEI != imei].sort_values("Timestamp")
            j = np.searchsorted(peer.Timestamp.to_numpy(dtype="datetime64[ns]"), target.Timestamp.to_numpy(dtype="datetime64[ns]"))
            left = np.clip(j - 1, 0, len(peer) - 1)
            right = np.clip(j, 0, len(peer) - 1)
            tn = target.Timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
            pn = peer.Timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
            chosen = np.where(np.abs(pn[right] - tn) < np.abs(pn[left] - tn), right, left)
            seconds = np.minimum(np.abs(pn[right] - tn), np.abs(pn[left] - tn)) / 1e9
            delta = target.env_residual.to_numpy(float) - peer.env_residual.to_numpy(float)[chosen]
            valid = seconds <= 1800
            idx = target.index.to_numpy()[valid]
            data.loc[idx, "peer_delta"] = delta[valid]
        valid = group.peer_delta.notna()
        if valid.sum() >= 8:
            vals = group.loc[valid, "peer_delta"].to_numpy(float)
            mad = np.median(np.abs(vals - np.median(vals))) * 1.4826 + 1e-9
            z = (vals - np.median(vals)) / mad
            data.loc[group.index[valid], "pseudo_anomaly_peer"] = (np.abs(z) >= 5).astype(int)
    data["pseudo_anomaly"] = ((data.pseudo_anomaly_individual == 1) | (data.pseudo_anomaly_peer == 1)).astype(int)
    return data


class TemporalDDPM(nn.Module):
    def __init__(self, features: int, hidden: int, steps: int):
        super().__init__()
        self.steps = steps
        self.time = nn.Embedding(steps, hidden)
        self.label = nn.Embedding(2, hidden)
        self.in_proj = nn.Conv1d(features, hidden, 1)
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv1d(hidden, features, 1),
        )
        beta = torch.linspace(1e-4, 0.02, steps)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0))

    def predict(self, x: torch.Tensor, step: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        c = (self.time(step) + self.label(label)).unsqueeze(-1)
        return self.net(h + c)

    def loss(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        b = len(x)
        step = torch.randint(0, self.steps, (b,), device=x.device)
        a = self.alpha_bar[step].view(b, 1, 1)
        noise = torch.randn_like(x)
        xt = a.sqrt() * x + (1 - a).sqrt() * noise
        return F.mse_loss(self.predict(xt, step, label), noise)

    @torch.no_grad()
    def sample(self, n: int, length: int, label: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(n, 3, length, device=device)
        y = torch.full((n,), label, dtype=torch.long, device=device)
        for s in reversed(range(self.steps)):
            step = torch.full((n,), s, dtype=torch.long, device=device)
            beta = self.beta[s]
            alpha = 1 - beta
            abar = self.alpha_bar[s]
            eps = self.predict(x, step, y)
            mean = (x - beta / torch.sqrt(1 - abar) * eps) / torch.sqrt(alpha)
            x = mean + (torch.sqrt(beta) * torch.randn_like(x) if s else 0)
        return x


def windows_for_group(data: pd.DataFrame, ids: np.ndarray, length: int, label: int) -> np.ndarray:
    chunks = []
    for imei, g in data.loc[ids].groupby("IMEI", sort=False):
        g = g.sort_values("Timestamp")
        values = g[PRIMARY].to_numpy(np.float32)
        labels = g.pseudo_anomaly.to_numpy(int)
        for start in range(0, len(g) - length + 1, max(1, length // 2)):
            if int(labels[start:start + length].max()) == label:
                chunks.append(values[start:start + length])
    return np.asarray(chunks, np.float32)


def fit_model(windows: np.ndarray, label: int, args, device: torch.device, seed: int) -> tuple[TemporalDDPM, StandardScaler]:
    scaler = StandardScaler().fit(windows.reshape(-1, 3))
    z = scaler.transform(windows.reshape(-1, 3)).reshape(windows.shape).transpose(0, 2, 1).astype(np.float32)
    model = TemporalDDPM(3, args.hidden, args.steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    tensor = torch.from_numpy(z).to(device)
    labels = torch.full((len(tensor),), label, dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(args.epochs):
        order = torch.randperm(len(tensor), generator=rng, device=device)
        for start in range(0, len(order), args.batch):
            sel = order[start:start + args.batch]
            loss = model.loss(tensor[sel], labels[sel])
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model, scaler


def generate_for_device(model, scaler, device, n, length, label, args):
    model.eval()
    chunks = []
    for start in range(0, n, args.batch):
        count = min(args.batch, n - start)
        z = model.sample(count, length, label, device).cpu().numpy().transpose(0, 2, 1)
        chunks.append(scaler.inverse_transform(z.reshape(-1, 3)).reshape(count, length, 3))
    return np.concatenate(chunks)[:n]


def save_long_plot(real: pd.DataFrame, generated: pd.DataFrame, out: Path, title: str) -> None:
    devices = list(real.IMEI.drop_duplicates())
    ncols, nrows = 5, int(np.ceil(len(devices) / 5))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 3.1 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, imei in zip(axes, devices):
        r = real[real.IMEI == imei]
        g = generated[generated.IMEI == imei]
        ax.plot(r.Timestamp, r.Measure, color="#64748b", lw=.7, alpha=.7)
        normal = r.pseudo_anomaly.eq(0)
        bad = r.pseudo_anomaly.eq(1)
        ax.scatter(r.loc[normal, "Timestamp"], r.loc[normal, "Measure"], color="#2563eb", s=5, alpha=.55, label="real normal")
        ax.scatter(r.loc[bad, "Timestamp"], r.loc[bad, "Measure"], color="#dc2626", s=12, alpha=.9, label="real candidate anomaly")
        if len(g):
            ax.plot(g.Timestamp, g.Measure, color="#111827", lw=.55, alpha=.65)
            gbad = g.generated_label.eq(1)
            ax.scatter(g.loc[~gbad, "Timestamp"], g.loc[~gbad, "Measure"], color="#0ea5e9", s=4, alpha=.45, label="generated normal")
            ax.scatter(g.loc[gbad, "Timestamp"], g.loc[gbad, "Measure"], color="#f97316", s=11, alpha=.9, label="generated candidate anomaly")
        ax.set_title(f"{imei}\nreal={int(bad.sum())} generated={int(g.generated_label.sum()) if len(g) else 0}", fontsize=8)
        ax.grid(alpha=.25)
        ax.tick_params(axis="x", labelrotation=30, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(devices):]: ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.01), fontsize=9)
    fig.suptitle(title, fontsize=15, y=1.035)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(args) -> None:
    seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = label_points(load_points(args.workbook))
    data.to_csv(args.output / "real_annotated_points.csv", index=False, encoding="utf-8-sig")
    generated_rows = []
    audit = []
    for group_id, group in data.groupby("group_id", sort=False):
        if pd.isna(group_id): continue
        group_ids = group.index.to_numpy()
        devices = list(group.IMEI.unique())
        for imei in devices:
            g = group[group.IMEI == imei].sort_values("Timestamp")
            cut = max(args.sequence_length, int(len(g) * args.train_fraction))
            fit_ids = g.index[:cut].to_numpy()
            test = g.iloc[cut:].copy()
            if len(test) == 0: continue
            group_fit_ids = []
            for peer_imei, peer_frame in group.groupby("IMEI", sort=False):
                peer_frame = peer_frame.sort_values("Timestamp")
                peer_cut = max(args.sequence_length, int(len(peer_frame) * args.train_fraction))
                group_fit_ids.extend(peer_frame.index[:peer_cut].tolist())
            group_fit_ids = np.asarray(group_fit_ids, dtype=int)
            models = {}
            for label in (0, 1):
                train_windows = windows_for_group(data, group_fit_ids, args.sequence_length, label)
                if len(train_windows) < args.min_train_sequences:
                    audit.append({"group_id": group_id, "IMEI": imei, "label": label, "status": "skipped", "n_train_sequences": len(train_windows)})
                    continue
                local = int(hashlib.sha256(f"{group_id}|{label}|{args.seed}".encode()).hexdigest()[:8], 16)
                models[label] = (*fit_model(train_windows, label, args, device, local), len(train_windows))
            if not models:
                continue
            interval = test.Timestamp.diff().dt.total_seconds().median()
            interval = interval if np.isfinite(interval) and interval > 0 else 3600.0
            timestamps = pd.date_range(test.Timestamp.iloc[0], periods=len(test), freq=pd.to_timedelta(interval, unit="s"))
            for start in range(0, len(test), args.sequence_length):
                stop = min(start + args.sequence_length, len(test))
                source_label = int(test.iloc[start:stop].pseudo_anomaly.max())
                label = source_label if source_label in models else next(iter(models))
                model, scaler, n_train = models[label]
                generated = generate_for_device(model, scaler, device, 1, args.sequence_length, label, args)[0]
                values = generated[:stop - start]
                for timestamp, row in zip(timestamps[start:stop], values):
                    generated_rows.append({"IMEI": str(imei), "Location": g.Location.iloc[0], "group_id": group_id, "Timestamp": timestamp, "Measure": row[0], "Temp": row[1], "Hum": row[2], "condition_label": label, "source_start": str(timestamps[start]), "label_model": label})
            audit.append({"group_id": group_id, "IMEI": imei, "label": "mixed_by_test_chunk", "status": "generated", "n_train_sequences_0": models.get(0, (None, None, 0))[2], "n_train_sequences_1": models.get(1, (None, None, 0))[2], "n_generated_points": len(test)})
    generated = pd.DataFrame(generated_rows)
    if len(generated):
        relabeled = label_points(generated.copy())
        generated["generated_label"] = relabeled["pseudo_anomaly"].to_numpy(int)
    generated.to_csv(args.output / "generated_long_points.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(args.output / "generation_audit.csv", index=False, encoding="utf-8-sig")
    save_long_plot(data, generated, args.output / "real_vs_generated_long_timeseries.png", "Continuous peer-group sequence generation: real vs generated")
    (args.output / "protocol.json").write_text(json.dumps({"source": str(args.workbook), "scope": "peer_group", "sequence_length": args.sequence_length, "features": PRIMARY, "train_fraction": args.train_fraction, "epochs": args.epochs, "steps": args.steps, "seed": args.seed, "device": str(device), "timestamp_policy": "per-device median test sampling interval from test start", "label_policy": "generated class condition; real labels recomputed by robust residual + peer rule"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--min-train-sequences", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())

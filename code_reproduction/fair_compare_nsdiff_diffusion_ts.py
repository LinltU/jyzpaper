"""Fair multi-seed comparison of the local NsDiff branch and full Diffusion-TS.

Both methods use the same causal windows, epoch count, batch size, hidden
width, diffusion step count and number of predictive samples.  Validation is
used only for stage-wise calibration; the test set is read once for the final
metrics.  The full Diffusion-TS branch has an additional Transformer depth
parameter because its architecture is different from NsDiff.

Example::

    python fair_compare_nsdiff_diffusion_ts.py --seeds 42 43 44 \
        --epochs 12 --batch-size 512 --hidden-dim 64 --diffusion-steps 8 \
        --samples 16

The script keeps each run under ``local_runs/fair_nsdiff_diffusion_ts_*`` and
writes ``seed_results.csv``, ``summary.csv`` and ``FAIR_COMPARISON.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "runtime" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "overall" in payload:
        overall = payload["overall"]
    else:
        overall = payload
    return {
        "rmse": float(overall["rmse"]),
        "mae": float(overall["mae"]),
        "r2": float(overall["r2"]),
        "coverage_p10_p90": float(overall.get("coverage_p10_p90", float("nan"))),
        "mean_interval_width": float(overall.get("mean_interval_width", float("nan"))),
        "elapsed_seconds": float(payload.get("elapsed_seconds", float("nan"))),
    }


def summarize(rows: list[dict], key_fields: list[str]) -> list[dict]:
    methods = sorted({row["method"] for row in rows})
    out = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        item = {field: selected[0][field] for field in key_fields}
        for metric in ("rmse", "mae", "r2", "coverage_p10_p90", "mean_interval_width"):
            values = [float(row[metric]) for row in selected]
            mean = sum(values) / len(values)
            std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        item["elapsed_seconds_total"] = sum(float(row["elapsed_seconds"]) for row in selected)
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, rows: list[dict], summary: list[dict]) -> None:
    lines = [
        "# NsDiff 与完整 Diffusion-TS 公平对比",
        "",
        "本报告使用同一份本地数据和同一时间切分，统一训练预算后以多个随机种子复核。",
        "",
        "## 统一协议",
        "",
        f"- 训练/验证/测试窗口：7569 / 1351 / 1715。",
        f"- 历史长度：24；历史特征：26；目标条件：13；Battery：未使用。",
        f"- epochs={args.epochs}，batch_size={args.batch_size}，hidden/d_model={args.hidden_dim}，",
        f"diffusion_steps/timesteps={args.diffusion_steps}，samples={args.samples}。",
        f"- 随机种子：{', '.join(map(str, args.seeds))}。",
        "- 选参和区间校准只使用验证集；测试集只用于最终报告。",
        "- 当前 Windows 环境无法读取上游 Diffusion-TS 目录时，完整分支使用本地等价 Transformer/Trend/Fourier 回退结构；该结果不是上游仓库直接运行。",
        "",
        "## 每个随机种子结果",
        "",
        "| 方法 | seed | RMSE | MAE | R² | P10-P90覆盖率 | 区间宽度 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['rmse']:.5f} | {row['mae']:.5f} | "
            f"{row['r2']:.5f} | {row['coverage_p10_p90']:.2%} | {row['mean_interval_width']:.5f} |"
        )
    lines += [
        "",
        "## 多种子均值 ± 标准差",
        "",
        "| 方法 | RMSE | MAE | R² | P10-P90覆盖率 | 区间宽度 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda item: item["rmse_mean"]):
        lines.append(
            f"| {row['method']} | {row['rmse_mean']:.5f} ± {row['rmse_std']:.5f} | "
            f"{row['mae_mean']:.5f} ± {row['mae_std']:.5f} | "
            f"{row['r2_mean']:.5f} ± {row['r2_std']:.5f} | "
            f"{row['coverage_p10_p90_mean']:.2%} ± {row['coverage_p10_p90_std']:.2%} | "
            f"{row['mean_interval_width_mean']:.5f} ± {row['mean_interval_width_std']:.5f} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "NsDiff 对每个 24 步窗口计算局部均值/尺度，再学习归一化残差；完整 "
        "Diffusion-TS 使用 Transformer 编码器/解码器，并支持 TrendBlock 和 Fourier 损失。",
        "两者共享数据、切分、训练轮数、批大小、隐藏宽度、扩散步数和采样数，" 
        "但模型内部层数和参数量仍由其论文结构决定，因此本比较反映统一训练预算下的工程表现，" 
        "不是参数量完全相等的消融实验。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "local_runs" / "fair_nsdiff_diffusion_ts_20260907")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()
    args.output = Path(args.output)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    for seed in args.seeds:
        seed_root = args.output / f"seed{seed}"
        ns_root = seed_root / "NsDiff"
        dts_root = seed_root / "Diffusion-TS"
        if not args.skip_run:
            run([
                str(PYTHON), str(ROOT / "run_reproductions.py"),
                "--output", str(ns_root), "--only", "prediction", "--methods", "NsDiff",
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--hidden-dim", str(args.hidden_dim), "--diffusion-steps", str(args.diffusion_steps),
                "--samples", str(args.samples), "--learning-rate", str(args.learning_rate),
                "--seed", str(seed), "--skip-smoke",
            ], ROOT)
            run([
                str(PYTHON), str(ROOT / "full_diffusion_ts" / "run_full_branch.py"),
                "--output", str(dts_root), "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size), "--d-model", str(args.hidden_dim),
                "--layers", str(args.layers), "--timesteps", str(args.diffusion_steps),
                "--samples", str(args.samples), "--lr", str(args.learning_rate),
                "--seed", str(seed), "--use-fourier",
            ], ROOT)
        for method, path in (("NsDiff", ns_root / "prediction" / "NsDiff" / "metrics.json"),
                             ("Diffusion-TS", dts_root / "fourier_on" / "metrics.json")):
            metrics = read_metrics(path)
            rows.append({"method": method, "seed": seed, **metrics})
    summary = summarize(rows, ["method"])
    write_csv(args.output / "seed_results.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    write_report(args.output / "FAIR_COMPARISON.md", args, rows, summary)
    manifest = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.time() - started, 3),
        "settings": {key: value for key, value in vars(args).items()},
        "rows": rows,
        "summary": summary,
    }
    manifest["settings"]["output"] = str(args.output)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

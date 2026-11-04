"""Repeat validation-selected prediction configurations across random seeds."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_reproductions.py"
PYTHON = ROOT / "runtime" / "python.exe"
CONFIGS = {
    "TimeGrad": (60, 160, 24, 5e-4, 120, 3),
    "DiffPLF": (60, 160, 24, 5e-4, 120, 3),
    "CCDM": (60, 160, 24, 5e-4, 120, 3),
    "TSDE": (60, 160, 24, 5e-4, 120, 3),
    "NsDiff": (60, 160, 24, 5e-4, 120, 3),
    "Diffusion-TS": (60, 160, 24, 5e-4, 120, 3),
    "Treeffuser": (60, 96, 16, 5e-4, 120, 3),
}

def main():
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "local_runs" / "best_seed_repeats_20260905"
    rows=[]
    for seed in (42,43,44):
        for method, (epochs, hidden, steps, lr, estimators, repeats) in CONFIGS.items():
            out = out_root / f"seed{seed}" / method
            cmd=[str(PYTHON if PYTHON.exists() else sys.executable),str(RUNNER),"--output",str(out),"--only","prediction","--methods",method,"--epochs",str(epochs),"--hidden-dim",str(hidden),"--diffusion-steps",str(steps),"--samples","32","--learning-rate",str(lr),"--tree-estimators",str(estimators),"--tree-repeats",str(repeats),"--seed",str(seed),"--skip-smoke"]
            print(f"[repeat] seed={seed} method={method}", flush=True)
            p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
            (out/"stdout_stderr.log").write_text(p.stdout+"\n"+p.stderr,encoding="utf-8")
            mp=out/"prediction"/method/"metrics.json"
            if p.returncode!=0 or not mp.exists():
                rows.append({"seed":seed,"method":method,"status":"failed","returncode":p.returncode}); continue
            j=json.loads(mp.read_text(encoding="utf-8")); v=j.get("validation_before_test_calibration",{}); t=j["overall"]
            rows.append({"seed":seed,"method":method,"status":"ok","validation_rmse":v.get("rmse"),"test_rmse":t.get("rmse"),"test_mae":t.get("mae"),"test_r2":t.get("r2"),"test_coverage_p10_p90":t.get("coverage_p10_p90"),"test_mean_interval_width":t.get("mean_interval_width")})
    df=pd.DataFrame(rows); out_root.mkdir(parents=True,exist_ok=True); df.to_csv(out_root/"seed_results.csv",index=False,encoding="utf-8-sig")
    ok=df[df.status=="ok"]
    if not ok.empty:
        summary=ok.groupby("method").agg({"test_rmse":["mean","std","min","max"],"test_mae":["mean","std"],"test_coverage_p10_p90":["mean","std"]}).reset_index()
        summary.columns = ["_".join(item).strip("_") if isinstance(item, tuple) else str(item) for item in summary.columns]
        summary.to_csv(out_root/"seed_summary.csv",index=False,encoding="utf-8-sig")
        (out_root/"seed_summary.json").write_text(json.dumps(summary.to_dict(orient="records"),ensure_ascii=False,indent=2),encoding="utf-8")

if __name__ == "__main__": main()

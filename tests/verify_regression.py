"""Compare portable rerun outputs to a privately held corrected v4 delivery."""
import argparse
import json
from pathlib import Path
import pandas as pd

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("--actual", type=Path, required=True)
cli.add_argument("--reference", type=Path, required=True)
args = cli.parse_args()
files = [
    "device_points_v4.csv", "window_features_v4.csv", "candidate_windows_v4.csv",
    "device_summary_v4.csv", "peer_matches_v4.csv", "source_cleaning_audit.csv",
    "peer_jump_events_v4.csv",
]
checked = {}
for name in files:
    actual = pd.read_csv(args.actual / name, low_memory=False)
    reference = pd.read_csv(args.reference / name, low_memory=False)
    pd.testing.assert_frame_equal(actual, reference, check_exact=False,
                                  rtol=1e-12, atol=1e-12)
    checked[name] = {"rows": len(actual), "columns": len(actual.columns)}
actual_audit = json.loads((args.actual / "audit_v4.json").read_text(encoding="utf-8"))
reference_audit = json.loads((args.reference / "audit_v4.json").read_text(encoding="utf-8"))
assert actual_audit == reference_audit
print(json.dumps({"status": "passed", "csvs": checked, "audit": "identical"}, indent=2))

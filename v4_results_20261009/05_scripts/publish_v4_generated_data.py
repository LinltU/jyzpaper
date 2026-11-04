"""Copy generated-data figures into the v4 collection, preserving originals.

Only visualization outputs and the three visualization scripts are published;
raw NPZ data and the existing consolidated report/PDF are not modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main(args):
    source = args.run.resolve() / "figures/generated_data"
    root = args.collection.resolve()
    target = root / "03_benchmark_v4_evidence_features/figures/generated_data"
    if not source.is_dir() or not (root / "05_scripts").is_dir() or not target.parent.is_dir():
        raise ValueError("Source figures or collection parent directories missing")
    protocol = json.loads((source / "plot_protocol.json").read_text())
    expected = protocol["configurations"] * (3 if protocol["details"] else 2)
    if len(list(source.glob("*/*/*.png"))) != expected:
        raise ValueError("Unexpected figure count; finish plotting before publishing")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    pairs = [(p, target / p.relative_to(source)) for p in files]
    pairs.extend((HERE / name, root / "05_scripts" / name) for name in (
        "plot_v4_generated_data.py", "test_plot_v4_generated_data.py", "publish_v4_generated_data.py"))
    records = []
    for src, dst in pairs:
        if src.resolve() == dst.resolve():
            raise ValueError("Publication source and destination must differ")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sha = digest(src)
        if digest(dst) != sha:
            raise IOError(f"Copy verification failed: {dst}")
        records.append({"source": str(src), "target": dst.relative_to(root).as_posix(),
                        "bytes": dst.stat().st_size, "sha256": sha})
    manifest = root / "06_reports/GENERATED_DATA_PUBLICATION_MANIFEST.json"
    manifest.write_text(json.dumps({"files": records, "figures": expected,
                                   "originals_preserved": True}, indent=2), encoding="utf-8")
    print(f"Published {expected} figures; verified {len(records)} files via SHA-256")
    print("Collection:", root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs/device_data_110_220_paper_aug_v4evi_20261009")
    parser.add_argument("--collection", type=Path, default=HERE.parents[1] / "v4_results_20261009")
    main(parser.parse_args())

#!/usr/bin/env python3
"""Fetch pinned GitHub repository snapshots into the reproduction workspace."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
MANIFEST = ROOT / "sources_manifest.json"

REPOSITORIES = {
    "TimeGrad-pytorch-ts": "zalandoresearch/pytorch-ts",
    "DiffPLF": "LSY-Cython/DiffPLF",
    "CCDM": "LSY-Cython/CCDM",
    "TSDE": "EQTPartners/TSDE",
    "NsDiff": "wwy155/NsDiff",
    "Treeffuser": "blei-lab/treeffuser",
    "CSDI": "ermongroup/CSDI",
    "SSSD": "ai4healthuol/sssd",
    "MTSCI": "JeremyChou28/MTSCI",
    "SSD-TS": "decisionintelligence/SSD-TS",
    "CoSTI": "javiersgjavi/CoSTI",
    "DiffAD": "ChunjingXiao/DiffAD",
    "D3R": "ForestsKing/D3R",
    "TIMEADDM": "Hurongyao/TIMEADDM",
    "IMDiffusion": "17000cyh/IMDiffusion",
    "ProDiffAD": "AiShare-WHU/ProDiffAD",
    "Diffusion-TS": "Y-debug-sys/Diffusion-TS",
}


def request_json(url: str, attempts: int = 3) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "JYZmentor-Reproduction",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=60
            ) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def download(url: str, target: Path, attempts: int = 3) -> None:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "JYZmentor-Reproduction",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=180
            ) as response, target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return
        except (OSError, urllib.error.URLError):
            if target.exists():
                target.unlink()
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)


def fetch(name: str, repository: str) -> dict:
    metadata = request_json(f"https://api.github.com/repos/{repository}")
    branch = metadata["default_branch"]
    commit = request_json(
        f"https://api.github.com/repos/{repository}/commits/{branch}"
    )["sha"]
    destination = UPSTREAM / name
    marker = destination / ".source.json"
    if marker.exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("commit") == commit:
            recorded["status"] = "present"
            return recorded

    with tempfile.TemporaryDirectory(prefix="jyzmentor_repo_") as temp_dir:
        archive = Path(temp_dir) / f"{name}.zip"
        download(
            f"https://api.github.com/repos/{repository}/zipball/{commit}", archive
        )
        extract_root = Path(temp_dir) / "extract"
        extract_root.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extract_root)
        children = [path for path in extract_root.iterdir() if path.is_dir()]
        if len(children) != 1:
            raise RuntimeError(f"Unexpected archive layout for {repository}")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(children[0]), destination)

    record = {
        "name": name,
        "repository": repository,
        "url": f"https://github.com/{repository}",
        "default_branch": branch,
        "commit": commit,
        "license": (metadata.get("license") or {}).get("spdx_id"),
        "stars_snapshot": metadata.get("stargazers_count"),
        "pushed_at": metadata.get("pushed_at"),
        "retrieval": "GitHub API zipball pinned to commit",
        "status": "downloaded",
    }
    marker.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> int:
    UPSTREAM.mkdir(parents=True, exist_ok=True)
    records = []
    for name, repository in REPOSITORIES.items():
        print(f"FETCH {name}: {repository}", flush=True)
        try:
            record = fetch(name, repository)
        except Exception as exc:  # Keep an auditable record of unavailable sources.
            record = {
                "name": name,
                "repository": repository,
                "url": f"https://github.com/{repository}",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"FAILED {name}: {exc}", file=sys.stderr, flush=True)
        records.append(record)
        MANIFEST.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    failed = [record for record in records if record.get("status") == "failed"]
    print(f"Completed: {len(records) - len(failed)}/{len(records)} sources", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

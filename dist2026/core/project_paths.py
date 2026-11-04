"""Portable path helpers for the JYZmentor project.

Runtime configuration files should store paths relative to the configuration
file that owns them. ``resolve_project_path`` also understands legacy absolute
paths containing ``dist2026`` or ``code reproduction``. This fallback keeps
old experiment manifests usable after the whole project directory is moved.
"""

from __future__ import annotations

import os
from pathlib import Path


DIST_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DIST_DIR.parent
PROJECT_MARKERS = ("dist2026", "code reproduction")


def _rebased_candidates(path: Path) -> list[Path]:
    """Rebase a legacy absolute project path below the current project root."""
    lowered = [part.lower() for part in path.parts]
    candidates = []
    for marker in PROJECT_MARKERS:
        marker_lower = marker.lower()
        if marker_lower in lowered:
            index = lowered.index(marker_lower)
            candidates.append(PROJECT_ROOT.joinpath(*path.parts[index:]))
    return candidates


def resolve_project_path(
    value: str | os.PathLike,
    *,
    relative_to: str | os.PathLike | None = None,
    must_exist: bool = True,
) -> Path:
    """Resolve relative and legacy absolute paths without fixing a drive/root.

    Relative values are first resolved next to the file that references them,
    then below ``dist2026`` and the repository root. A missing legacy absolute
    path is rebased from its ``dist2026``/``code reproduction`` component.
    """
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    raw = Path(expanded)
    base = Path(relative_to).resolve() if relative_to is not None else Path.cwd()

    if raw.is_absolute():
        candidates = [raw, *_rebased_candidates(raw)]
    else:
        candidates = [base / raw, DIST_DIR / raw, PROJECT_ROOT / raw]

    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
        if resolved.exists():
            return resolved

    if must_exist:
        attempted = "\n  - ".join(str(path) for path in unique)
        raise FileNotFoundError(
            "Project path does not exist. Tried:\n  - " + attempted
        )
    return unique[0]


def manifest_relative_path(
    target: str | os.PathLike,
    manifest_path: str | os.PathLike,
) -> str:
    """Return a portable POSIX-style path relative to a manifest directory."""
    target_path = Path(target).resolve()
    manifest_dir = Path(manifest_path).resolve().parent
    return Path(os.path.relpath(target_path, manifest_dir)).as_posix()

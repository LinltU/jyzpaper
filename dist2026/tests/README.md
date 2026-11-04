# Current tests

This directory contains the maintained tests for the current dist2026 pipeline.

Run from the repository root:

```powershell
& "C:\Project\JYZmentor\scripts\run_tests.ps1"
```

Use `-Full` to run every `test_*.py` file in this directory. Archived legacy
comparison tests are under `dist2026/archive/legacy_tests/` and are not part of
the default verification path.

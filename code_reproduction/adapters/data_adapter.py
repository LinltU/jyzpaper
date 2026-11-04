"""Data adapter for paper reproductions on the JYZmentor sensor data.

The adapter deliberately keeps the task causal: a forecast window contains only
rows strictly before the target timestamp. Target temperature, humidity,
location and line voltage are treated as known covariates; target Measure-derived
features are never passed to a forecasting model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIST_ROOT = PROJECT_ROOT / "dist2026"
if str(DIST_ROOT) not in sys.path:
    sys.path.insert(0, str(DIST_ROOT))

from core.feature_engineering_v2 import prepare_rows_v2  # noqa: E402


RAW_COLUMNS = ["IMEI", "Timestamp", "Battery", "Measure", "Temp", "Hum", "Version", "Location"]

# A compact, causal set of measured/derived channels.  Battery is deliberately
# absent because it is the measurement device state, not an insulator voltage
# signal.
HISTORY_COLUMNS = [
    "Measure",
    "Temp",
    "Hum",
    "AH",
    "DeltaHours",
    "MeasureRate",
    "TempRate",
    "HumRate",
    "MeasureEMA3",
    "HumEMA3",
    "MeasureMAD6",
    "HumMAD6",
    "HoursAbove65",
    "HoursAbove80",
    "HumRising",
    "HumFalling",
    "HumInvalid",
    "LineVoltageKV",
    "Loc_SideCode",
    "Loc_PhaseCode",
    "Loc_SizeCode",
    "Loc_StrNum",
    "Loc_PlateCount",
    "Loc_PlateStart",
    "Loc_PlateEnd",
    "Loc_PlatePositionRatio",
]

# Known at the target timestamp in an online setting.  Measure-derived target
# features are intentionally excluded to prevent target leakage.
TARGET_CONTEXT_COLUMNS = [
    "Temp",
    "Hum",
    "AH",
    "DeltaHours",
    "LineVoltageKV",
    "Loc_SideCode",
    "Loc_PhaseCode",
    "Loc_SizeCode",
    "Loc_StrNum",
    "Loc_PlateCount",
    "Loc_PlateStart",
    "Loc_PlateEnd",
    "Loc_PlatePositionRatio",
]

STAGES = ("dry", "transition", "humid", "very_humid")


def stage_name(humidity: float) -> str:
    if humidity < 50.0:
        return "dry"
    if humidity < 65.0:
        return "transition"
    if humidity < 80.0:
        return "humid"
    return "very_humid"


def read_data(project_root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_root = project_root / "dist2026" / "latest_retraining_data_20260716"
    train = pd.read_csv(data_root / "training_merged.csv")
    test = pd.read_csv(data_root / "test_latest_holdout.csv")
    combined = pd.concat([train, test], ignore_index=True)
    combined["IMEI"] = combined["IMEI"].astype(str)
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"], errors="raise")
    combined = combined.sort_values(["IMEI", "Timestamp"]).reset_index(drop=True)
    return train, test, combined


def prepare_data(combined: pd.DataFrame) -> pd.DataFrame:
    # prepare_rows_v2 is the canonical feature implementation already used by
    # the project. It also performs causal rate/EMA/MAD construction by device.
    return prepare_rows_v2(combined.copy())


def _key_series(frame: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    return set(zip(frame["IMEI"].astype(str), pd.to_datetime(frame["Timestamp"])))


def build_forecast_windows(
    prepared: pd.DataFrame,
    target_keys: set[tuple[str, pd.Timestamp]],
    seq_len: int = 24,
) -> dict[str, np.ndarray | list[dict]]:
    """Create causal one-step windows for the selected target rows."""
    rows = []
    for device_id, group in prepared.groupby("IMEI", sort=False):
        group = group.sort_values("Timestamp").reset_index(drop=True)
        if len(group) <= seq_len:
            continue
        history = group[HISTORY_COLUMNS].to_numpy(dtype=np.float32)
        for target_index in range(seq_len, len(group)):
            target_row = group.iloc[target_index]
            key = (str(device_id), pd.Timestamp(target_row["Timestamp"]))
            if key not in target_keys:
                continue
            previous = history[target_index - seq_len : target_index]
            context = target_row[TARGET_CONTEXT_COLUMNS].to_numpy(dtype=np.float32)
            rows.append(
                {
                    "x": previous,
                    "context": context,
                    "target": float(target_row["Measure"]),
                    "persistence": float(group.iloc[target_index - 1]["Measure"]),
                    "IMEI": str(device_id),
                    "Timestamp": pd.Timestamp(target_row["Timestamp"]).isoformat(),
                    "Temp": float(target_row["Temp"]),
                    "Hum": float(target_row["Hum"]),
                    "stage": stage_name(float(target_row["Hum"])),
                    "Location": str(target_row["Location"]),
                }
            )
    if not rows:
        raise ValueError("No forecast windows were created")
    return {
        "x": np.stack([row["x"] for row in rows]).astype(np.float32),
        "context": np.stack([row["context"] for row in rows]).astype(np.float32),
        "target": np.asarray([row["target"] for row in rows], dtype=np.float32),
        "persistence": np.asarray([row["persistence"] for row in rows], dtype=np.float32),
        "metadata": [
            {key: value for key, value in row.items() if key not in {"x", "context", "target", "persistence"}}
            for row in rows
        ],
    }


def split_train_validation(
    windows: dict[str, np.ndarray | list[dict]], validation_ratio: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    metadata = windows["metadata"]
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for device_id in sorted({item["IMEI"] for item in metadata}):
        indices = [index for index, item in enumerate(metadata) if item["IMEI"] == device_id]
        count = max(1, int(np.ceil(len(indices) * validation_ratio)))
        train_indices.extend(indices[:-count])
        validation_indices.extend(indices[-count:])
    return np.asarray(train_indices, dtype=np.int64), np.asarray(validation_indices, dtype=np.int64)


def fit_scaler(windows: dict[str, np.ndarray | list[dict]], indices: np.ndarray) -> dict[str, np.ndarray | float]:
    x = windows["x"][indices]
    context = windows["context"][indices]
    target = windows["target"][indices]
    x_mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    x_std = x.reshape(-1, x.shape[-1]).std(axis=0)
    context_mean = context.mean(axis=0)
    context_std = context.std(axis=0)
    target_mean = float(target.mean())
    target_std = float(target.std())
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    context_std = np.where(context_std < 1e-6, 1.0, context_std).astype(np.float32)
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std,
        "context_mean": context_mean.astype(np.float32),
        "context_std": context_std,
        "target_mean": target_mean,
        "target_std": max(target_std, 1e-6),
    }


def transform_windows(windows: dict[str, np.ndarray | list[dict]], scaler: dict) -> dict:
    x = (windows["x"] - scaler["x_mean"]) / scaler["x_std"]
    context = (windows["context"] - scaler["context_mean"]) / scaler["context_std"]
    target = (windows["target"] - scaler["target_mean"]) / scaler["target_std"]
    persistence = (windows["persistence"] - scaler["target_mean"]) / scaler["target_std"]
    return {
        **windows,
        "x": x.astype(np.float32),
        "context": context.astype(np.float32),
        "target": target.astype(np.float32),
        "persistence": persistence.astype(np.float32),
    }


def inverse_target(values: np.ndarray, scaler: dict) -> np.ndarray:
    return values * float(scaler["target_std"]) + float(scaler["target_mean"])


def device_indices(metadata: list[dict], train_metadata: list[dict]) -> tuple[np.ndarray, dict[str, int]]:
    devices = sorted({item["IMEI"] for item in train_metadata})
    mapping = {device_id: index for index, device_id in enumerate(devices)}
    unknown = len(mapping)
    return np.asarray([mapping.get(item["IMEI"], unknown) for item in metadata], dtype=np.int64), mapping


def stage_indices(metadata: list[dict]) -> np.ndarray:
    return np.asarray([STAGES.index(item["stage"]) for item in metadata], dtype=np.int64)


def make_tabular_features(
    windows: dict[str, np.ndarray | list[dict]],
    use_history: int = 6,
    use_device_one_hot: bool = True,
    device_map: dict[str, int] | None = None,
) -> np.ndarray:
    """Flatten a small causal context for Treeffuser-style tabular diffusion."""
    x = windows["x"][:, -use_history:, :]
    context = windows["context"]
    parts = [x.reshape(len(x), -1), context]
    if use_device_one_hot:
        if device_map is None:
            device_map = {device: index for index, device in enumerate(sorted({m["IMEI"] for m in windows["metadata"]}))}
        one_hot = np.zeros((len(x), len(device_map)), dtype=np.float32)
        for row, metadata in enumerate(windows["metadata"]):
            if metadata["IMEI"] in device_map:
                one_hot[row, device_map[metadata["IMEI"]]] = 1.0
        parts.append(one_hot)
    return np.concatenate(parts, axis=1).astype(np.float32)

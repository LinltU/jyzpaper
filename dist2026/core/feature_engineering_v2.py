"""Causal, compact feature engineering for humidity-stage voltage prediction."""

import re

import numpy as np
import pandas as pd


CORE_COLUMNS_V2 = [
    "Measure",
    "Temp",
    "Hum",
    "AH",
    "DeltaHours",
    "DeviceElapsedDays",
    "Hour_sin",
    "Hour_cos",
]

DYNAMIC_COLUMNS_V2 = [
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
]

LOCATION_COLUMNS_V2 = [
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

FEATURE_COLUMNS_V2_COMPACT = CORE_COLUMNS_V2 + LOCATION_COLUMNS_V2
FEATURE_COLUMNS_V2 = CORE_COLUMNS_V2 + DYNAMIC_COLUMNS_V2 + LOCATION_COLUMNS_V2

CLEAN_TIME_COLUMNS = [
    "Hour_sin",
    "Hour_cos",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "Month_sin",
    "Month_cos",
]

CLEAN_CORE_COLUMNS = [
    "Measure",
    "Temp",
    "Hum",
    "AH",
    "DeltaHours",
    "DeviceElapsedDays",
] + CLEAN_TIME_COLUMNS + LOCATION_COLUMNS_V2


def absolute_humidity(temp, hum):
    saturation = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
    vapor_pressure = hum / 100.0 * saturation
    return 216.7 * vapor_pressure / (temp + 273.15)


def _require_parsed(values, location, field_name):
    if values.isna().any():
        bad_locations = sorted(location[values.isna()].unique().tolist())
        raise ValueError(
            "Could not parse {} from Location: {}".format(
                field_name,
                ", ".join(bad_locations),
            )
        )
    return values.astype(float)


def _chinese_integer(value):
    if not isinstance(value, str) or not value:
        return np.nan
    digits = {
        "\u96f6": 0,
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
    }
    if "\u5341" not in value:
        return float(digits.get(value, np.nan))
    before, after = value.split("\u5341", 1)
    tens = digits.get(before, 1) if before else 1
    ones = digits.get(after, 0) if after else 0
    return float(tens * 10 + ones)


def parse_location_features(location):
    """Parse compact and Chinese location labels into physical attributes."""
    location = location.fillna("").astype(str).str.strip()
    head = location.str.split("-").str[0]

    left = head.str.match(r"^L", case=False) | location.str.contains("left|\u5de6", case=False)
    right = head.str.match(r"^R", case=False) | location.str.contains("right|\u53f3", case=False)
    upper = head.str.contains("Up", case=False, regex=False) | location.str.contains("upper|\u4e0a\u76f8", case=False)
    lower = head.str.contains("Low", case=False, regex=False) | location.str.contains("lower|\u4e0b\u76f8", case=False)
    small = head.str.contains("SStr", case=False, regex=False) | location.str.contains("small|\u5c0f\u53f7\u4fa7", case=False)
    big = head.str.contains("BStr", case=False, regex=False) | location.str.contains("big|\u5927\u53f7\u4fa7", case=False)

    parsed = pd.DataFrame(index=location.index)
    parsed["Loc_SideCode"] = np.select([left, right], [-1.0, 1.0], default=0.0)
    parsed["Loc_PhaseCode"] = np.select([lower, upper], [-1.0, 1.0], default=0.0)
    parsed["Loc_SizeCode"] = np.select([small, big], [-1.0, 1.0], default=0.0)

    string_number = location.str.extract(r"(?:Str|\u4e32)\s*(\d+)", flags=re.IGNORECASE, expand=False)
    parsed["Loc_StrNum"] = _require_parsed(
        pd.to_numeric(string_number, errors="coerce"),
        location,
        "string number",
    )

    line_voltage = location.str.extract(
        r"(\d+(?:\.\d+)?)\s*kV",
        flags=re.IGNORECASE,
        expand=False,
    )
    parsed["LineVoltageKV"] = _require_parsed(
        pd.to_numeric(line_voltage, errors="coerce"),
        location,
        "line voltage",
    )

    plate_count = location.str.extract(
        r"kV[-_](\d+)(?:[-_]|\s*\u7247)",
        flags=re.IGNORECASE,
        expand=False,
    )
    chinese_plate_count = location.str.extract(r"(\d+)\s*\u7247", expand=False)
    plate_count = plate_count.fillna(chinese_plate_count)
    parsed["Loc_PlateCount"] = _require_parsed(
        pd.to_numeric(plate_count, errors="coerce"),
        location,
        "plate count",
    )

    plate_range = location.str.extract(
        r"(?:^|[-_])P[-_](\d+)[-_~\uFF5E](\d+)$",
        flags=re.IGNORECASE,
        expand=True,
    )
    generic_range = location.str.extract(
        r"(?:\u7b2c)?(\d+)\s*(?:-|~|\uFF5E|\u81f3)\s*(\d+)\s*(?:\u7247)?$",
        expand=True,
    )
    plate_range = plate_range.fillna(generic_range)
    chinese_range = location.str.extract(
        r"(?:\u7b2c)?([\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+)\s*"
        r"(?:-|~|\uFF5E|\u81f3)\s*"
        r"([\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+)\s*(?:\u7247)?$",
        expand=True,
    )
    chinese_range[0] = chinese_range[0].map(_chinese_integer)
    chinese_range[1] = chinese_range[1].map(_chinese_integer)
    plate_range = plate_range.fillna(chinese_range)
    parsed["Loc_PlateStart"] = _require_parsed(
        pd.to_numeric(plate_range[0], errors="coerce"),
        location,
        "plate range start",
    )
    parsed["Loc_PlateEnd"] = _require_parsed(
        pd.to_numeric(plate_range[1], errors="coerce"),
        location,
        "plate range end",
    )
    midpoint = (parsed["Loc_PlateStart"] + parsed["Loc_PlateEnd"]) / 2.0
    parsed["Loc_PlatePositionRatio"] = midpoint / parsed["Loc_PlateCount"].clip(lower=1.0)
    return parsed


def _rolling_mad(series, window):
    def median_absolute_deviation(values):
        median = np.median(values)
        return np.median(np.abs(values - median))

    return series.rolling(window=window, min_periods=2).apply(
        median_absolute_deviation,
        raw=True,
    ).fillna(0.0)


def _consecutive_hours(result, threshold):
    durations = np.zeros(len(result), dtype=float)
    for positions in result.groupby("IMEI", sort=False).indices.values():
        positions = np.asarray(positions)
        humidity = result.iloc[positions]["Hum"].to_numpy(dtype=float)
        delta = result.iloc[positions]["DeltaHours"].to_numpy(dtype=float)
        for offset in range(1, len(positions)):
            if humidity[offset] >= threshold:
                previous = durations[positions[offset - 1]] if humidity[offset - 1] >= threshold else 0.0
                durations[positions[offset]] = previous + delta[offset]
    return durations


def prepare_rows_v2(
    df,
    humidity_fill_value=None,
    median_delta_hours=None,
    device_start_times=None,
):
    """Build causal features without changing the source dataframe."""
    required = {"IMEI", "Location", "Timestamp", "Measure", "Temp", "Hum"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    result = df.copy()
    result["IMEI"] = result["IMEI"].astype(str)
    result["Timestamp"] = pd.to_datetime(result["Timestamp"], errors="raise")
    result = result.sort_values(["IMEI", "Timestamp"]).reset_index(drop=True)

    invalid_humidity = (~np.isfinite(result["Hum"])) | (result["Hum"] <= 0.0) | (result["Hum"] > 100.0)
    result["HumInvalid"] = invalid_humidity.astype(float)
    clean_humidity = result["Hum"].mask(invalid_humidity)
    clean_humidity = clean_humidity.groupby(result["IMEI"], sort=False).ffill()
    if humidity_fill_value is None:
        humidity_fill_value = float(clean_humidity.median())
    if not np.isfinite(humidity_fill_value):
        humidity_fill_value = 50.0
    result["Hum"] = clean_humidity.fillna(humidity_fill_value).clip(0.0, 100.0)

    delta_hours = result.groupby("IMEI", sort=False)["Timestamp"].diff().dt.total_seconds() / 3600.0
    if median_delta_hours is None:
        median_delta_hours = float(delta_hours[delta_hours > 0.0].median())
    if not np.isfinite(median_delta_hours):
        median_delta_hours = 12.0
    result["DeltaHours"] = delta_hours.fillna(median_delta_hours).clip(1.0 / 60.0, 72.0)
    if device_start_times is None:
        device_start = result.groupby("IMEI", sort=False)["Timestamp"].transform("min")
    else:
        normalized_starts = {
            str(device_id): pd.Timestamp(timestamp)
            for device_id, timestamp in device_start_times.items()
        }
        device_start = result["IMEI"].map(normalized_starts)
        fallback_start = result.groupby("IMEI", sort=False)["Timestamp"].transform("min")
        device_start = device_start.fillna(fallback_start)
    result["DeviceElapsedDays"] = (
        result["Timestamp"] - device_start
    ).dt.total_seconds() / 86400.0

    result["AH"] = absolute_humidity(result["Temp"], result["Hum"])
    hour = result["Timestamp"].dt.hour + result["Timestamp"].dt.minute / 60.0
    result["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    day_of_week = result["Timestamp"].dt.dayofweek
    month = result["Timestamp"].dt.month
    result["DayOfWeek_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    result["DayOfWeek_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    result["Month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    result["Month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    location_features = parse_location_features(result["Location"])
    for column in location_features.columns:
        result[column] = location_features[column]

    safe_delta = result["DeltaHours"].clip(lower=1.0 / 60.0)
    rate_limits = {
        "Measure": ("MeasureRate", 1.0),
        "Temp": ("TempRate", 10.0),
        "Hum": ("HumRate", 25.0),
    }
    for source, (destination, limit) in rate_limits.items():
        difference = result.groupby("IMEI", sort=False)[source].diff().fillna(0.0)
        result[destination] = (difference / safe_delta).clip(-limit, limit)

    result["MeasureEMA3"] = result.groupby("IMEI", sort=False)["Measure"].transform(
        lambda series: series.ewm(span=3, adjust=False).mean()
    )
    result["HumEMA3"] = result.groupby("IMEI", sort=False)["Hum"].transform(
        lambda series: series.ewm(span=3, adjust=False).mean()
    )
    result["MeasureMAD6"] = result.groupby("IMEI", sort=False)["Measure"].transform(
        lambda series: _rolling_mad(series, window=6)
    )
    result["HumMAD6"] = result.groupby("IMEI", sort=False)["Hum"].transform(
        lambda series: _rolling_mad(series, window=6)
    )
    result["HoursAbove65"] = _consecutive_hours(result, 65.0)
    result["HoursAbove80"] = _consecutive_hours(result, 80.0)
    result["HumRising"] = (result["HumRate"] > 0.0).astype(float)
    result["HumFalling"] = (result["HumRate"] < 0.0).astype(float)

    numeric = result[FEATURE_COLUMNS_V2].replace([np.inf, -np.inf], np.nan)
    if numeric.isna().any().any():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError("Non-finite Feature V2 values: " + ", ".join(bad_columns))
    result[FEATURE_COLUMNS_V2] = numeric
    return result


def feature_columns_v2(profile="dynamic"):
    if profile == "compact":
        return list(FEATURE_COLUMNS_V2_COMPACT)
    if profile == "dynamic":
        return [
            column for column in FEATURE_COLUMNS_V2
            if column != "DeviceElapsedDays"
        ]
    if profile == "dynamic_elapsed":
        return list(FEATURE_COLUMNS_V2)
    raise ValueError("Unknown Feature V2 profile: " + str(profile))


def add_device_one_hot(result, device_ids):
    result = result.copy()
    for device_id in device_ids:
        result["Device_" + device_id] = (result["IMEI"] == device_id).astype(float)
    return result


def feature_columns_clean(device_ids):
    return list(CLEAN_CORE_COLUMNS) + ["Device_" + device_id for device_id in device_ids]


def feature_diagnostics(result, columns):
    constants = [column for column in columns if result[column].nunique(dropna=False) <= 1]
    return {
        "row_count": int(len(result)),
        "feature_count": int(len(columns)),
        "feature_columns": list(columns),
        "constant_features": constants,
        "humidity_invalid_rows": int(result["HumInvalid"].sum()) if "HumInvalid" in result else 0,
        "line_voltage_values": sorted(result["LineVoltageKV"].unique().tolist())
        if "LineVoltageKV" in result else [],
        "location_count": int(result["Location"].nunique()),
    }

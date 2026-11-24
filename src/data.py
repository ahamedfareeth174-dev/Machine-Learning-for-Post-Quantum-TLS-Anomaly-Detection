"""Dataset loading and split helpers."""

from __future__ import annotations

import pandas as pd

from .paths import META_COLS, ML_FEATURES_CSV, SCENARIO_MANIFEST_CSV


def load_ml_frame(path=None) -> pd.DataFrame:
    path = path or ML_FEATURES_CSV
    data = pd.read_csv(path)
    required = META_COLS
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Unexpected ML schema; missing columns: {sorted(missing)}")
    if not data["ID"].is_unique:
        raise ValueError("Expected one row per ID")
    if set(data["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Expected split values: train, val, test")
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("Expected binary labels {0, 1}")
    return data


def load_manifest(path=None) -> pd.DataFrame:
    path = path or SCENARIO_MANIFEST_CSV
    manifest = pd.read_csv(path).copy()
    manifest["join_id"] = manifest["ID"].str.replace(r"\.pcap$", "", regex=True)
    if not manifest["join_id"].is_unique:
        raise ValueError("Manifest join IDs are not unique")
    return manifest


def feature_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in META_COLS]


def split_xy(data: pd.DataFrame):
    cols = feature_columns(data)
    X = data[cols].astype(float)
    y = data["label"].astype(int)
    masks = {s: data["split"].eq(s) for s in ("train", "val", "test")}
    return cols, X, y, masks

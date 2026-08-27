from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureDataset:
    """The subset of readGripData's saved FeaturePool needed for decoding."""

    X: np.ndarray
    labels: pd.DataFrame
    windows: pd.DataFrame
    channel_names: tuple[str, ...]
    feature_names: tuple[str, ...]

    def validate(self) -> None:
        if self.X.ndim != 3:
            raise ValueError("X must have shape (windows, channels, features)")
        if len(self.labels) != len(self.X) or len(self.windows) != len(self.X):
            raise ValueError("features, labels and windows are not row-aligned")
        if self.X.shape[1] != len(self.channel_names):
            raise ValueError("channel metadata does not match X")
        if self.X.shape[2] != len(self.feature_names):
            raise ValueError("feature metadata does not match X")
        for column in ("window_id", "trial_key", "label_time_s"):
            if column not in self.windows:
                raise KeyError(f"windows.parquet is missing {column!r}")
        if not np.array_equal(self.windows["window_id"].to_numpy(), np.arange(len(self.X))):
            raise ValueError("windows.window_id is not aligned with X")


def load_feature_dataset(path: str | Path) -> FeatureDataset:
    """Load the five-file feature pool emitted by readGripData."""
    path = Path(path)
    required = ("features.npz", "labels.parquet", "windows.parquet", "feature_names.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Feature pool is missing: {missing}")
    with np.load(path / "features.npz", allow_pickle=False) as arrays:
        X = np.asarray(arrays["X"])
    info = json.loads((path / "feature_names.json").read_text(encoding="utf-8"))
    dataset = FeatureDataset(
        X=X,
        labels=pd.read_parquet(path / "labels.parquet"),
        windows=pd.read_parquet(path / "windows.parquet"),
        channel_names=tuple(item["name"] for item in info["channel_axis"]),
        feature_names=tuple(item["name"] for item in info["feature_axis"]),
    )
    dataset.validate()
    return dataset

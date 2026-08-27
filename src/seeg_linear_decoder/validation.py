from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .io import FeatureDataset


@dataclass(frozen=True)
class DecoderConfig:
    model: str = "ridge"
    target: str = "force_normalized"
    n_splits: int = 5
    alpha: float = 1.0
    l1_ratio: float = 0.5
    max_iter: int = 20_000
    tol: float = 1e-4
    n_permutations: int = 200
    min_shift_fraction: float = 0.2
    random_state: int = 0

    def validate(self) -> None:
        if self.model not in {"ols", "ridge", "lasso", "elasticnet"}:
            raise ValueError("model must be one of: ols, ridge, lasso, elasticnet")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.model in {"lasso", "elasticnet"} and self.alpha <= 0:
            raise ValueError("Lasso and Elastic Net require alpha > 0")
        if self.model == "ridge" and self.alpha < 0:
            raise ValueError("Ridge requires alpha >= 0")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must lie between 0 and 1")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.n_permutations < 0:
            raise ValueError("n_permutations must be non-negative")
        if not 0 < self.min_shift_fraction < 0.5:
            raise ValueError("min_shift_fraction must lie between 0 and 0.5")


def _pearson(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1])


def _model(config: DecoderConfig):
    if config.model == "ols":
        estimator = LinearRegression()
    elif config.model == "ridge":
        estimator = Ridge(alpha=config.alpha)
    elif config.model == "lasso":
        estimator = Lasso(
            alpha=config.alpha, max_iter=config.max_iter, tol=config.tol,
            random_state=config.random_state,
        )
    else:
        estimator = ElasticNet(
            alpha=config.alpha, l1_ratio=config.l1_ratio,
            max_iter=config.max_iter, tol=config.tol,
            random_state=config.random_state,
        )
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def _oof_predict(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds, config: DecoderConfig
) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=float)
    for train, test in folds:
        fitted = _model(config).fit(X[train], y[train])
        pred[test] = fitted.predict(X[test])
    return pred


def _circular_shift_within_groups(
    y: np.ndarray, groups: np.ndarray, rng: np.random.Generator, minimum_fraction: float
) -> np.ndarray:
    shifted = y.copy()
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups == group)
        n = len(idx)
        minimum = max(1, int(np.ceil(n * minimum_fraction)))
        if n < 2 * minimum + 1:
            shifted[idx] = rng.permutation(y[idx])
        else:
            offset = int(rng.integers(minimum, n - minimum + 1))
            shifted[idx] = np.roll(y[idx], offset)
    return shifted


def _select_rows(
    data: FeatureDataset,
    time_intervals: pd.DataFrame | None,
    base_mask: str | None,
) -> np.ndarray:
    keep = np.ones(len(data.X), dtype=bool)
    if base_mask:
        source = data.windows if base_mask in data.windows else data.labels
        if base_mask not in source:
            raise KeyError(f"Unknown mask column {base_mask!r}")
        keep &= source[base_mask].to_numpy(dtype=bool)
    if time_intervals is not None:
        required = {"trial_key", "start_s", "end_s"}
        if not required.issubset(time_intervals.columns):
            raise KeyError(f"time intervals require columns {sorted(required)}")
        interval_keep = np.zeros(len(data.X), dtype=bool)
        times = data.windows["label_time_s"].to_numpy(float)
        trials = data.windows["trial_key"].astype(str).to_numpy()
        selected = time_intervals
        if "include" in selected:
            selected = selected.loc[selected["include"].astype(bool)]
        for row in selected.itertuples(index=False):
            start, end = float(row.start_s), float(row.end_s)
            if end <= start:
                raise ValueError(f"Invalid interval for {row.trial_key}: end_s <= start_s")
            interval_keep |= (trials == str(row.trial_key)) & (times >= start) & (times < end)
        keep &= interval_keep
    return keep


def _validate_coordinates(
    channel_names: Sequence[str], coordinates: pd.DataFrame | None
) -> pd.DataFrame:
    if coordinates is None:
        return pd.DataFrame({"channel": channel_names})
    if "channel" not in coordinates:
        raise KeyError("coordinates CSV requires a 'channel' column")
    if coordinates["channel"].duplicated().any():
        raise ValueError("coordinates contains duplicate channel names")
    indexed = coordinates.set_index("channel", drop=False)
    missing = [name for name in channel_names if name not in indexed.index]
    if missing:
        raise ValueError(f"Selected channels have no coordinate row: {missing}")
    return indexed.loc[list(channel_names)].reset_index(drop=True)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p_values[valid])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def validate_channels(
    data: FeatureDataset,
    *,
    channels: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    coordinates: pd.DataFrame | None = None,
    time_intervals: pd.DataFrame | None = None,
    base_mask: str | None = "mask_flight",
    config: DecoderConfig = DecoderConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one selected linear decoder per channel with trial-held-out CV.

    Returns a channel summary and row-level out-of-fold predictions. The null
    p-value uses circular target shifts within each trial so slow force and
    neural autocorrelation are not destroyed.
    """
    data.validate()
    config.validate()
    if config.target not in data.labels:
        raise KeyError(f"Unknown target {config.target!r}")
    channel_names = list(channels or data.channel_names)
    unknown_channels = sorted(set(channel_names) - set(data.channel_names))
    if unknown_channels:
        raise KeyError(f"Unknown channels: {unknown_channels}")
    coord_rows = _validate_coordinates(channel_names, coordinates)
    feature_names = list(features or data.feature_names)
    unknown_features = sorted(set(feature_names) - set(data.feature_names))
    if unknown_features:
        raise KeyError(f"Unknown features: {unknown_features}")

    keep = _select_rows(data, time_intervals, base_mask)
    y = data.labels.loc[keep, config.target].to_numpy(float)
    groups = data.windows.loc[keep, "trial_key"].astype(str).to_numpy()
    finite_y = np.isfinite(y)
    y, groups = y[finite_y], groups[finite_y]
    n_groups = len(pd.unique(groups))
    if n_groups < config.n_splits:
        raise ValueError(f"Need at least {config.n_splits} trials after selection; found {n_groups}")
    splitter = GroupKFold(n_splits=config.n_splits)
    dummy = np.zeros((len(y), 1))
    folds = list(splitter.split(dummy, y, groups))
    rng = np.random.default_rng(config.random_state)
    shifted_targets = [
        _circular_shift_within_groups(y, groups, rng, config.min_shift_fraction)
        for _ in range(config.n_permutations)
    ]
    f_idx = [data.feature_names.index(name) for name in feature_names]

    summaries, predictions = [], []
    for channel in channel_names:
        c_idx = data.channel_names.index(channel)
        X = data.X[keep][:, c_idx, :][:, f_idx][finite_y]
        pred = _oof_predict(X, y, groups, folds, config)
        observed_r = _pearson(y, pred)
        null_r = np.asarray([
            _pearson(y_null, _oof_predict(X, y_null, groups, folds, config))
            for y_null in shifted_targets
        ])
        valid_null = null_r[np.isfinite(null_r)]
        p_value = (
            float((1 + np.sum(valid_null >= observed_r)) / (1 + len(valid_null)))
            if np.isfinite(observed_r) else float("nan")
        )
        summary = {
            "channel": channel,
            "model": config.model,
            "alpha": 0.0 if config.model == "ols" else config.alpha,
            "l1_ratio": config.l1_ratio if config.model == "elasticnet" else float("nan"),
            "n_windows": len(y),
            "n_trials": n_groups,
            "n_features": len(feature_names),
            "pearson_r": observed_r,
            "r2": float(r2_score(y, pred)),
            "mae": float(mean_absolute_error(y, pred)),
            "permutation_p": p_value,
            "null_r_mean": float(np.nanmean(null_r)) if len(null_r) else float("nan"),
        }
        coord = coord_rows.loc[coord_rows["channel"] == channel].iloc[0].to_dict()
        summary.update({key: value for key, value in coord.items() if key != "channel"})
        summaries.append(summary)
        predictions.append(pd.DataFrame({
            "channel": channel,
            "model": config.model,
            "trial_key": groups,
            "y_true": y,
            "y_pred": pred,
        }))
    summary_df = pd.DataFrame(summaries)
    summary_df["permutation_q_fdr_bh"] = _benjamini_hochberg(
        summary_df["permutation_p"].to_numpy(float)
    )
    summary_df = summary_df.sort_values("pearson_r", ascending=False)
    return summary_df.reset_index(drop=True), pd.concat(predictions, ignore_index=True)

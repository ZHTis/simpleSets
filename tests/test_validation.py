import numpy as np
import pandas as pd

from seeg_linear_decoder import DecoderConfig, FeatureDataset, validate_channels


def test_effective_channel_ranks_first():
    rng = np.random.default_rng(12)
    n_trials, per_trial = 8, 50
    groups = np.repeat([f"T{i}" for i in range(n_trials)], per_trial)
    latent = rng.normal(size=n_trials * per_trial)
    X = rng.normal(size=(len(groups), 2, 3))
    X[:, 0, 0] = latent + rng.normal(scale=0.05, size=len(groups))
    y = 2.0 * latent + rng.normal(scale=0.1, size=len(groups))
    ids = np.arange(len(groups))
    data = FeatureDataset(
        X=X,
        labels=pd.DataFrame({"window_id": ids, "force_normalized": y}),
        windows=pd.DataFrame({
            "window_id": ids,
            "trial_key": groups,
            "label_time_s": np.tile(np.arange(per_trial) / 10, n_trials),
            "mask_flight": True,
        }),
        channel_names=("A1", "A2"),
        feature_names=("f0", "f1", "f2"),
    )
    summary, pred = validate_channels(
        data,
        config=DecoderConfig(n_splits=4, n_permutations=9, random_state=3),
    )
    assert summary.iloc[0]["channel"] == "A1"
    assert summary.iloc[0]["pearson_r"] > 0.95
    assert len(pred) == 2 * len(y)


def test_intervals_and_coordinates_are_applied():
    rng = np.random.default_rng(1)
    groups = np.repeat([f"T{i}" for i in range(4)], 10)
    ids = np.arange(40)
    data = FeatureDataset(
        X=rng.normal(size=(40, 1, 1)),
        labels=pd.DataFrame({"window_id": ids, "force_normalized": rng.normal(size=40)}),
        windows=pd.DataFrame({
            "window_id": ids, "trial_key": groups,
            "label_time_s": np.tile(np.arange(10), 4), "mask_flight": True,
        }),
        channel_names=("A1",), feature_names=("f",),
    )
    intervals = pd.DataFrame({"trial_key": [f"T{i}" for i in range(4)], "start_s": 2, "end_s": 7})
    coords = pd.DataFrame({"channel": ["A1"], "x": [1.2], "y": [2.3], "z": [3.4]})
    summary, _ = validate_channels(
        data, coordinates=coords, time_intervals=intervals,
        config=DecoderConfig(n_splits=2, n_permutations=0),
    )
    assert summary.iloc[0]["n_windows"] == 20
    assert summary.iloc[0]["x"] == 1.2


def test_all_four_model_branches_run():
    rng = np.random.default_rng(22)
    groups = np.repeat([f"T{i}" for i in range(6)], 20)
    X = rng.normal(size=(len(groups), 1, 4))
    y = 1.5 * X[:, 0, 0] + rng.normal(scale=0.2, size=len(groups))
    ids = np.arange(len(groups))
    data = FeatureDataset(
        X=X,
        labels=pd.DataFrame({"window_id": ids, "force_normalized": y}),
        windows=pd.DataFrame({
            "window_id": ids, "trial_key": groups,
            "label_time_s": np.tile(np.arange(20), 6), "mask_flight": True,
        }),
        channel_names=("A1",), feature_names=("f0", "f1", "f2", "f3"),
    )
    settings = {
        "ols": {},
        "ridge": {"alpha": 1.0},
        "lasso": {"alpha": 0.01},
        "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5},
    }
    for model, kwargs in settings.items():
        summary, predictions = validate_channels(
            data,
            config=DecoderConfig(
                model=model, n_splits=3, n_permutations=0, **kwargs,
            ),
        )
        assert summary.iloc[0]["model"] == model
        assert predictions["model"].unique().tolist() == [model]
        assert summary.iloc[0]["pearson_r"] > 0.9

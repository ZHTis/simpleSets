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


def test_all_ten_model_branches_run():
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
        "glm": {"alpha": 0.01, "glm_family": "normal"},
        "spline": {"alpha": 0.01, "spline_n_knots": 4},
        "tree": {"tree_max_depth": 4, "tree_min_samples_leaf": 5},
        "bayesian": {},
        "autoregressive": {"alpha": 0.1, "ar_lags": 2},
        "mlp": {
            "mlp_hidden_layer_sizes": (8,), "mlp_early_stopping": False,
            "mlp_max_iter": 1_000,
        },
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


def test_bayesian_regression_exports_predictive_uncertainty():
    rng = np.random.default_rng(44)
    groups = np.repeat([f"T{i}" for i in range(6)], 20)
    x = rng.normal(size=len(groups))
    y = 1.2 * x + rng.normal(scale=0.25, size=len(groups))
    ids = np.arange(len(groups))
    data = FeatureDataset(
        X=x.reshape(-1, 1, 1),
        labels=pd.DataFrame({"window_id": ids, "force_normalized": y}),
        windows=pd.DataFrame({
            "window_id": ids, "trial_key": groups,
            "label_time_s": np.tile(np.arange(20), 6), "mask_flight": True,
        }),
        channel_names=("A1",), feature_names=("f0",),
    )
    summary, predictions = validate_channels(
        data,
        config=DecoderConfig(
            model="bayesian", n_splits=3, n_permutations=0,
        ),
    )
    assert np.isfinite(predictions["y_pred_std"]).all()
    assert (predictions["y_pred_std"] > 0).all()
    assert 0 <= summary.iloc[0]["predictive_interval_95_coverage"] <= 1


def test_autoregressive_lags_do_not_cross_trial_boundaries():
    from seeg_linear_decoder.validation import _lagged_targets

    y = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    groups = np.asarray(["T1", "T1", "T1", "T2", "T2", "T2"])
    lagged = _lagged_targets(y, groups, n_lags=2)
    assert np.isnan(lagged[0]).all()
    assert np.isnan(lagged[3]).all()
    assert lagged[4, 0] == 10.0
    assert np.isnan(lagged[4, 1])
    assert lagged[5].tolist() == [20.0, 10.0]


def test_glm_target_domain_is_checked_before_fitting():
    groups = np.repeat(["T0", "T1"], 4)
    ids = np.arange(len(groups))
    data = FeatureDataset(
        X=np.arange(len(groups), dtype=float).reshape(-1, 1, 1),
        labels=pd.DataFrame({"window_id": ids, "force_normalized": np.arange(-4, 4)}),
        windows=pd.DataFrame({
            "window_id": ids, "trial_key": groups,
            "label_time_s": np.tile(np.arange(4), 2), "mask_flight": True,
        }),
        channel_names=("A1",), feature_names=("f0",),
    )
    try:
        validate_channels(
            data,
            config=DecoderConfig(
                model="glm", glm_family="poisson", n_splits=2, n_permutations=0,
            ),
        )
    except ValueError as exc:
        assert "non-negative target" in str(exc)
    else:
        raise AssertionError("Poisson GLM accepted a negative target")


def test_positive_target_glm_families_run():
    rng = np.random.default_rng(31)
    groups = np.repeat([f"T{i}" for i in range(4)], 15)
    x = rng.normal(size=len(groups))
    y = np.exp(0.3 * x) + 0.1
    ids = np.arange(len(groups))
    data = FeatureDataset(
        X=x.reshape(-1, 1, 1),
        labels=pd.DataFrame({"window_id": ids, "force_normalized": y}),
        windows=pd.DataFrame({
            "window_id": ids, "trial_key": groups,
            "label_time_s": np.tile(np.arange(15), 4), "mask_flight": True,
        }),
        channel_names=("A1",), feature_names=("f0",),
    )
    for family in ("poisson", "gamma", "inverse_gaussian", "tweedie"):
        summary, predictions = validate_channels(
            data,
            config=DecoderConfig(
                model="glm", glm_family=family, glm_power=1.5,
                alpha=0.01, n_splits=2, n_permutations=0,
            ),
        )
        assert summary.iloc[0]["glm_family"] == family
        assert np.isfinite(predictions["y_pred"]).all()


def test_interpretable_nonlinear_models_capture_curvature():
    rng = np.random.default_rng(8)
    n_trials, per_trial = 8, 50
    groups = np.repeat([f"T{i}" for i in range(n_trials)], per_trial)
    x = rng.uniform(-2, 2, size=len(groups))
    X = x.reshape(-1, 1, 1)
    y = x ** 2 + rng.normal(scale=0.05, size=len(groups))
    ids = np.arange(len(groups))
    data = FeatureDataset(
        X=X,
        labels=pd.DataFrame({"window_id": ids, "force_normalized": y}),
        windows=pd.DataFrame({
            "window_id": ids, "trial_key": groups,
            "label_time_s": np.tile(np.arange(per_trial), n_trials), "mask_flight": True,
        }),
        channel_names=("A1",), feature_names=("f0",),
    )
    for model, kwargs in {
        "spline": {"spline_n_knots": 5},
        "tree": {"tree_max_depth": 4, "tree_min_samples_leaf": 8},
    }.items():
        summary, _ = validate_channels(
            data,
            config=DecoderConfig(
                model=model, alpha=0.01, n_splits=4, n_permutations=0, **kwargs,
            ),
        )
        assert summary.iloc[0]["r2"] > 0.8

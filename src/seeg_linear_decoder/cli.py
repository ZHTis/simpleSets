from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import load_feature_dataset
from .validation import DecoderConfig, validate_channels


def _csv_list(value: str | None):
    return None if not value else [item.strip() for item in value.split(",") if item.strip()]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate sEEG channels with trial-held-out linear decoding")
    p.add_argument("feature_pool", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--coordinates", type=Path, help="CSV with channel and optional x,y,z columns")
    p.add_argument("--time-labels", type=Path, help="CSV with trial_key,start_s,end_s[,include]")
    p.add_argument("--channels", help="Comma-separated channel names; default: all")
    p.add_argument("--features", help="Comma-separated feature names; default: all")
    p.add_argument("--target", default="force_normalized")
    p.add_argument(
        "--model", choices=("ols", "ridge", "lasso", "elasticnet"), default="ridge",
        help="Linear decoder branch; default: ridge",
    )
    p.add_argument("--mask", default="mask_flight", help="Boolean window/label column; use 'none' to disable")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--l1-ratio", type=float, default=0.5, help="Elastic Net L1 share; default: 0.5")
    p.add_argument("--max-iter", type=int, default=20_000)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = parser().parse_args()
    data = load_feature_dataset(args.feature_pool)
    coordinates = pd.read_csv(args.coordinates) if args.coordinates else None
    intervals = pd.read_csv(args.time_labels) if args.time_labels else None
    config = DecoderConfig(
        model=args.model,
        target=args.target,
        n_splits=args.folds,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        tol=args.tol,
        n_permutations=args.permutations,
        random_state=args.seed,
    )
    summary, predictions = validate_channels(
        data,
        channels=_csv_list(args.channels),
        features=_csv_list(args.features),
        coordinates=coordinates,
        time_intervals=intervals,
        base_mask=None if args.mask.lower() == "none" else args.mask,
        config=config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "channel_summary.csv", index=False)
    predictions.to_parquet(args.output_dir / "oof_predictions.parquet", index=False)
    run = vars(args).copy()
    run.update({"feature_pool": str(args.feature_pool), "output_dir": str(args.output_dir),
                "coordinates": str(args.coordinates) if args.coordinates else None,
                "time_labels": str(args.time_labels) if args.time_labels else None})
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

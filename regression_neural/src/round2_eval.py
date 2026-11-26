"""Rolling-origin comparisons. Save row-level local predictions for paired analysis."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_fold_features
from modeling import fit_predict_forward_lgb
from run_pipeline import select_cols
from utils import EXP_DIR, load_data, save_submission


RESULTS = Path(EXP_DIR) / "round2"
FOLDS = {
    "early": ("2023-01-01", "2023-07-01"),
    "dev": ("2023-07-01", "2024-07-01"),
    "follow": ("2024-07-01", "2025-07-01"),
}
ORIGINS = {**FOLDS, "jan24": ("2024-01-01", "2025-01-01")}


def split(train, fold):
    start, stop = ORIGINS[fold]
    past = train.loc[train.dt < start].reset_index(drop=True)
    future = train.loc[(train.dt >= start) & (train.dt < stop)].reset_index(drop=True)
    assert past.dt.max() < future.dt.min()
    assert future.groupby(future.dt.dt.normalize()).size().eq(48).all()
    return past, future


def record(name, fold, frame, pred):
    RESULTS.mkdir(parents=True, exist_ok=True)
    assert np.isfinite(pred).all() and len(pred) == len(frame)
    output = frame[["id", "datetime_utc", "generation_mw", "installed_capacity_mwp"]].copy()
    output["prediction"] = pred
    output.to_csv(RESULTS / f"{name}_{fold}.csv.gz", index=False)
    err = pred - output.generation_mw
    print(f"{name} {fold} MAE={err.abs().mean():.6f} bias={err.mean():+.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alignment", choices=["start", "end"])
    parser.add_argument("folds", nargs="+", choices=[*ORIGINS, "full"])
    args = parser.parse_args()
    train, test, template = load_data()
    for fold in args.folds:
        past, future = (train, test) if fold == "full" else split(train, fold)
        x, night, xe, ne, groups = build_fold_features(past, future, True, args.alignment)
        pred, rounds, _ = fit_predict_forward_lgb(
            x, night, past, xe, ne, future, select_cols(groups),
            mode="residual", add_phys_feat=True,
        )
        print(f"{fold} rounds={rounds}", flush=True)
        if fold == "full":
            path = RESULTS / f"m2_{args.alignment}_reproduction.csv"
            _, digest = save_submission(test, pred, path, template)
            if args.alignment == "end":
                assert digest == "06896f7f3a9a930545c6985971af64cdde8531855218062b76fb2944b1b1f846"
        else:
            record("m2_" + args.alignment, fold, future, pred)


if __name__ == "__main__":
    main()

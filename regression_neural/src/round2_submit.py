"""Regenerate the frozen round-2 predictions from the three shipped CSVs."""
import argparse
from pathlib import Path

import numpy as np

from features import build_fold_features
from round2_models import predict
from utils import DEFAULT_DATA_DIR, EXP_DIR, SUB_DIR, load_data, save_submission


SELECTED = "blend_family"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--replay", action="store_true",
                        help="Write an independent audit copy in experiments/round2.")
    args = parser.parse_args()
    train, test, sample = load_data(args.data_dir)
    assert len(train) == 61296 and len(test) == len(sample) == 17520
    assert train.dt.max() < test.dt.min()
    for part in (train, test):
        assert part.id.is_unique and part.dt.diff().dropna().dt.total_seconds().eq(1800).all()
        assert part.groupby(part.dt.dt.normalize()).size().eq(48).all()
    assert sample.id.is_unique and set(sample.id) == set(test.id)
    x, night, xe, ne = build_fold_features(train, test)
    pred = predict(SELECTED, x, night, train, xe, ne, test)
    assert np.isfinite(pred).all()
    path = (Path(EXP_DIR)/"round2"/"round2_replay.csv" if args.replay
            else Path(SUB_DIR)/"submission_round2.csv")
    save_submission(test, pred, path, sample)


if __name__ == "__main__":
    main()

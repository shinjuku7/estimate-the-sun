import csv
import hashlib
import json
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DATA_DIR = os.path.join(ROOT_DIR, "data")
# Prefer data inside a standalone checkout. Keep the existing workspace layout.
WORKSPACE_DATA_DIR = os.path.join(os.path.dirname(ROOT_DIR), "data")
DEFAULT_DATA_DIR = WORKSPACE_DATA_DIR if (
    not os.path.isdir(REPO_DATA_DIR) and os.path.isdir(WORKSPACE_DATA_DIR)
) else REPO_DATA_DIR
SUB_DIR = os.path.join(ROOT_DIR, "submissions")
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
MONTHLY_DIR = os.path.join(EXP_DIR, "monthly")
LOG_CSV_PATH = os.path.join(EXP_DIR, "log.csv")

CITIES = [
    "london", "birmingham", "manchester", "leeds", "bristol", "norwich",
    "southampton", "plymouth", "cardiff", "newcastle", "glasgow", "edinburgh"
]
VARS = ["ghi", "clearsky", "temp_c", "cloud_pct", "wind_ms", "rh_pct"]

# Exact duplicate columns in NASA POWER grid (bristol & cardiff share same radiation/cloud grid cell)
DUP_COLS = ["cardiff_ghi", "cardiff_clearsky", "cardiff_cloud_pct"]

LOG_COLUMNS = [
    "timestamp_utc",
    "exp_name",
    "split_role",
    "n_features",
    "params_json",
    "seed",
    "best_iteration",
    "dev_mae",
    "confirm_mae",
    "mean_bias_mw",
    "monthly_mae_path",
    "confirm_run_count",
    "notes",
]


def resolve_data_file(data_dir, canonical_name, fallback_name):
    """Look for canonical filename (e.g. train.csv) first, then fallback (solar-britain-train.csv)."""
    p1 = os.path.join(data_dir, canonical_name)
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(data_dir, fallback_name)
    if os.path.isfile(p2):
        return p2
    raise FileNotFoundError(f"Could not find {canonical_name} or {fallback_name} in {data_dir}")


def load_data(data_dir=DEFAULT_DATA_DIR):
    """Load train, test, and sample_submission from data_dir, sorted chronologically by datetime_utc."""
    train_path = resolve_data_file(data_dir, "train.csv", "solar-britain-train.csv")
    test_path = resolve_data_file(data_dir, "test.csv", "solar-britain-test.csv")
    sub_path = resolve_data_file(data_dir, "sample_submission.csv", "solar-britain-sample_submission.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sub_path)

    train["dt"] = pd.to_datetime(train["datetime_utc"], utc=True)
    test["dt"] = pd.to_datetime(test["datetime_utc"], utc=True)

    train = train.sort_values("dt").reset_index(drop=True)
    test = test.sort_values("dt").reset_index(drop=True)

    train["cf"] = train["generation_mw"] / train["installed_capacity_mwp"]
    return train, test, sub


def get_forward_splits(train):
    """
    Chronological validation splits:
    - dev:     Train 2022-01..2023-06 (1.5y), Eval 2023-07..2024-06 (1.0y)
    - confirm: Train 2022-01..2024-06 (2.5y), Eval 2024-07..2025-06 (1.0y)
    """
    dt = train["dt"]
    t_dev_end = pd.Timestamp("2023-07-01 00:00:00+00:00")
    t_conf_end = pd.Timestamp("2024-07-01 00:00:00+00:00")
    t_train_end = pd.Timestamp("2025-07-01 00:00:00+00:00")

    dev_tr_idx = np.where(dt < t_dev_end)[0]
    dev_va_idx = np.where((dt >= t_dev_end) & (dt < t_conf_end))[0]

    conf_tr_idx = np.where(dt < t_conf_end)[0]
    conf_va_idx = np.where((dt >= t_conf_end) & (dt < t_train_end))[0]

    return {
        "dev": (dev_tr_idx, dev_va_idx),
        "confirm": (conf_tr_idx, conf_va_idx),
    }


def split_internal_es_days(dt_series, val_frac=0.15, seed=42):
    """Deterministic whole-day split (about 15% of calendar dates) within a training fold for early stopping, hashed by (seed, YYYY-MM-DD). Returns boolean mask is_es_val."""
    date_strs = dt_series.dt.strftime("%Y-%m-%d").values
    unique_dates = np.unique(date_strs)
    threshold = int(round(val_frac * 10000))
    val_dates = set()
    for d in unique_dates:
        h = hashlib.sha256(f"es_day_split:{seed}:{d}".encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) % 10000
        if bucket < threshold:
            val_dates.add(d)
    is_es_val = np.isin(date_strs, list(val_dates))
    return is_es_val


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))))


def mean_bias(y_true, y_pred):
    """Signed mean bias: mean(pred - true) in MW. Positive = over-prediction."""
    return float(np.mean(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_monthly_metrics(dt_series, y_true, y_pred, exp_name, split_role):
    """Compute and save monthly MAE and mean bias CSV; return relative path."""
    os.makedirs(MONTHLY_DIR, exist_ok=True)
    df_m = pd.DataFrame({
        "ym": dt_series.dt.strftime("%Y-%m").values,
        "y_true": np.asarray(y_true, dtype=np.float64),
        "y_pred": np.asarray(y_pred, dtype=np.float64),
    })
    df_m["abs_err"] = np.abs(df_m["y_pred"] - df_m["y_true"])
    df_m["bias"] = df_m["y_pred"] - df_m["y_true"]
    out = df_m.groupby("ym", as_index=False).agg(
        n_rows=("abs_err", "count"),
        mae_mw=("abs_err", "mean"),
        mean_bias_mw=("bias", "mean"),
        mean_true_mw=("y_true", "mean"),
        mean_pred_mw=("y_pred", "mean"),
    )
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in f"{exp_name}_{split_role}")
    out_path = os.path.join(MONTHLY_DIR, f"{safe_name}.csv")
    out.to_csv(out_path, index=False, float_format="%.4f")
    rel_path = os.path.relpath(out_path, ROOT_DIR).replace("\\", "/")
    return rel_path, out


def get_confirm_run_count():
    """Return how many times confirm_mae has been logged in experiments/log.csv."""
    if not os.path.isfile(LOG_CSV_PATH):
        return 0
    df = pd.read_csv(LOG_CSV_PATH)
    if "confirm_mae" not in df.columns:
        return 0
    return int(df["confirm_mae"].notna().sum())


def log_experiment(
    exp_name,
    split_role,
    n_features,
    params_dict,
    seed,
    best_iteration,
    dev_mae=None,
    confirm_mae=None,
    mean_bias_mw=None,
    monthly_mae_path="",
    notes="",
):
    """Append an experiment record to experiments/log.csv."""
    os.makedirs(EXP_DIR, exist_ok=True)
    file_exists = os.path.isfile(LOG_CSV_PATH)

    prev_confirm_runs = get_confirm_run_count()
    confirm_run_count = prev_confirm_runs + (1 if confirm_mae is not None else 0)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exp_name": exp_name,
        "split_role": split_role,
        "n_features": int(n_features),
        "params_json": json.dumps(params_dict, sort_keys=True),
        "seed": int(seed),
        "best_iteration": "" if best_iteration is None else int(best_iteration),
        "dev_mae": "" if dev_mae is None else f"{dev_mae:.6f}",
        "confirm_mae": "" if confirm_mae is None else f"{confirm_mae:.6f}",
        "mean_bias_mw": "" if mean_bias_mw is None else f"{mean_bias_mw:.6f}",
        "monthly_mae_path": monthly_mae_path,
        "confirm_run_count": int(confirm_run_count),
        "notes": notes,
    }

    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return confirm_run_count


def save_submission(test_df, preds_mw, out_path, sub_template):
    """Save submission matching exact sample_submission ID order, run basic checks, and return (path, sha256)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    pred_map = pd.Series(np.asarray(preds_mw, dtype=np.float64), index=test_df["id"].values)
    sub = pd.DataFrame({
        "id": sub_template["id"].values,
        "generation_mw": pred_map.loc[sub_template["id"].values].values,
    })

    assert len(sub) == 17520, f"Expected 17520 rows, got {len(sub)}"
    assert list(sub.columns) == ["id", "generation_mw"], f"Bad columns: {list(sub.columns)}"
    assert sub["generation_mw"].isna().sum() == 0, "NaNs found in submission"
    assert np.isfinite(sub["generation_mw"].values).all(), "Inf found in submission"
    assert (sub["generation_mw"].values >= 0.0).all(), "Negative predictions found"
    assert (sub["id"].values == sub_template["id"].values).all(), "ID order mismatch"

    sub.to_csv(out_path, index=False, float_format="%.4f", lineterminator="\n")
    digest = file_sha256(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[SAVED] {out_path} ({size_mb:.2f} MB) | mean={sub['generation_mw'].mean():.2f} MW, "
          f"max={sub['generation_mw'].max():.2f} MW, zero_frac={(sub['generation_mw'] == 0).mean():.4f} | sha256={digest}")
    return out_path, digest

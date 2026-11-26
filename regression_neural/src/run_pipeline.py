import argparse
import os
import time
import numpy as np

from utils import (
    DEFAULT_DATA_DIR,
    SUB_DIR,
    load_data,
    get_forward_splits,
    split_internal_es_days,
    mae,
    mean_bias,
    save_monthly_metrics,
    log_experiment,
    save_submission,
)
from features import build_fold_features
from modeling import (
    DEFAULT_SEED,
    DEFAULT_NUM_THREADS,
    fit_predict_forward_lgb,
)

# Keep the Ridge/LightGBM residual stack and use only supplied weather and timestamps.
FROZEN_GROUPS = ["base", "temporal", "spatial", "regime"]
FROZEN_MODE = "residual"
FROZEN_ADD_PHYS_FEAT = True
FROZEN_RIDGE_ALPHA = 500.0


def select_cols(groups_dict):
    cols = []
    for g in FROZEN_GROUPS:
        cols.extend(groups_dict[g])
    return cols


def main():
    parser = argparse.ArgumentParser(
        description="Train M2, evaluate chronological folds, and generate the test submission."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory with train/test/sample_submission CSVs.")
    parser.add_argument(
        "--output",
        default=os.path.join(SUB_DIR, "sub_m2_no_coordinates.csv"),
        help="Path to write submission CSV.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--exp-name", default="M2_no_coordinates")
    parser.add_argument("--notes", default="Coordinate-free Ridge and LightGBM residual stack")
    args = parser.parse_args()

    np.random.seed(args.seed)
    t0 = time.time()

    print("=== Solar Britain M2 Pipeline ===")
    print(f"data_dir={args.data_dir} | seed={args.seed} | num_threads={args.num_threads}")
    print(f"Frozen config: groups={FROZEN_GROUPS}, mode={FROZEN_MODE}, add_phys_feat={FROZEN_ADD_PHYS_FEAT}\n")

    train, test, sub_template = load_data(args.data_dir)
    splits = get_forward_splits(train)

    # 1. DEV FOLD (2022-01..2023-06 -> 2023-07..2024-06)
    dev_tr_idx, dev_va_idx = splits["dev"]
    df_dev_tr = train.iloc[dev_tr_idx].reset_index(drop=True)
    df_dev_va = train.iloc[dev_va_idx].reset_index(drop=True)
    X_dev_tr, night_dev_tr, X_dev_va, night_dev_va, groups = build_fold_features(
        df_dev_tr, df_dev_va, return_groups=True
    )
    feat_cols = select_cols(groups)
    n_feats_total = len(feat_cols) + (1 if FROZEN_ADD_PHYS_FEAT else 0)

    pred_dev, it_dev, params_dev = fit_predict_forward_lgb(
        X_dev_tr, night_dev_tr, df_dev_tr,
        X_dev_va, night_dev_va, df_dev_va,
        feat_cols,
        mode=FROZEN_MODE,
        add_phys_feat=FROZEN_ADD_PHYS_FEAT,
        daytime_only=True,
        seed=args.seed,
        num_threads=args.num_threads,
        ridge_alpha=FROZEN_RIDGE_ALPHA,
    )
    dev_mae = mae(df_dev_va["generation_mw"].values, pred_dev)
    dev_bias = mean_bias(df_dev_va["generation_mw"].values, pred_dev)
    dev_mpath, _ = save_monthly_metrics(df_dev_va["dt"], df_dev_va["generation_mw"].values, pred_dev, args.exp_name, "dev")
    print(f"[DEV FOLD    ] (2023-07..2024-06) iter={it_dev:4d} | MAE={dev_mae:7.3f} MW | Bias={dev_bias:+7.3f} MW | monthly={dev_mpath}")

    # 2. CONFIRM FOLD (2022-01..2024-06 -> 2024-07..2025-06)
    conf_tr_idx, conf_va_idx = splits["confirm"]
    df_conf_tr = train.iloc[conf_tr_idx].reset_index(drop=True)
    df_conf_va = train.iloc[conf_va_idx].reset_index(drop=True)
    is_es_conf = split_internal_es_days(df_conf_tr["dt"], val_frac=0.15, seed=args.seed)
    n_days_c = df_conf_tr["dt"].dt.date.nunique()
    n_es_c = df_conf_tr.loc[is_es_conf, "dt"].dt.date.nunique()

    X_conf_tr, night_conf_tr, X_conf_va, night_conf_va = build_fold_features(
        df_conf_tr, df_conf_va, return_groups=False
    )
    pred_conf, it_conf, params_conf = fit_predict_forward_lgb(
        X_conf_tr, night_conf_tr, df_conf_tr,
        X_conf_va, night_conf_va, df_conf_va,
        feat_cols,
        mode=FROZEN_MODE,
        add_phys_feat=FROZEN_ADD_PHYS_FEAT,
        daytime_only=True,
        seed=args.seed,
        num_threads=args.num_threads,
        ridge_alpha=FROZEN_RIDGE_ALPHA,
    )
    conf_mae = mae(df_conf_va["generation_mw"].values, pred_conf)
    conf_bias = mean_bias(df_conf_va["generation_mw"].values, pred_conf)
    conf_mpath, conf_monthly_df = save_monthly_metrics(
        df_conf_va["dt"], df_conf_va["generation_mw"].values, pred_conf, args.exp_name, "confirm"
    )
    log_params = {
        **params_conf,
        "groups": FROZEN_GROUPS,
        "mode": FROZEN_MODE,
        "add_phys_feat": FROZEN_ADD_PHYS_FEAT,
        "ridge_alpha": FROZEN_RIDGE_ALPHA,
        "dev_best_iter": it_dev,
        "confirm_best_iter": it_conf,
    }
    conf_run_count = log_experiment(
        exp_name=args.exp_name,
        split_role="dev_and_confirm",
        n_features=n_feats_total,
        params_dict=log_params,
        seed=args.seed,
        best_iteration=it_conf,
        dev_mae=dev_mae,
        confirm_mae=conf_mae,
        mean_bias_mw=conf_bias,
        monthly_mae_path=conf_mpath,
        notes=f"{args.notes} | confirm_es_days={n_es_c}/{n_days_c}",
    )
    print(f"[CONFIRM FOLD] (2024-07..2025-06) iter={it_conf:4d} | MAE={conf_mae:7.3f} MW | Bias={conf_bias:+7.3f} MW | "
          f"es_days={n_es_c}/{n_days_c} ({n_es_c/n_days_c*100:.2f}%) | confirm_run_count={conf_run_count} | monthly={conf_mpath}")
    print("\nConfirm Fold Monthly Breakdown:")
    print(conf_monthly_df.to_string(index=False))

    # 3. FULL TRAIN -> TEST SUBMISSION (2022-01..2025-06 -> 2025-07..2026-06)
    is_es_full = split_internal_es_days(train["dt"], val_frac=0.15, seed=args.seed)
    n_days_f = train["dt"].dt.date.nunique()
    n_es_f = train.loc[is_es_full, "dt"].dt.date.nunique()

    X_full_tr, night_full_tr, X_test, night_test = build_fold_features(
        train, test, return_groups=False
    )
    pred_test, it_full, _ = fit_predict_forward_lgb(
        X_full_tr, night_full_tr, train,
        X_test, night_test, test,
        feat_cols,
        mode=FROZEN_MODE,
        add_phys_feat=FROZEN_ADD_PHYS_FEAT,
        daytime_only=True,
        seed=args.seed,
        num_threads=args.num_threads,
        ridge_alpha=FROZEN_RIDGE_ALPHA,
    )
    print(f"\n[FINAL MODEL ] (2022-01..2025-06) iter={it_full:4d} | es_days={n_es_f}/{n_days_f} ({n_es_f/n_days_f*100:.2f}%)")
    out_path, digest = save_submission(test, pred_test, args.output, sub_template)
    print(f"SUBMISSION_SHA256={digest}")
    print(f"Total pipeline runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

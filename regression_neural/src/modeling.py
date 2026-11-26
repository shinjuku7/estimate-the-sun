import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from utils import split_internal_es_days
from features import RAD_CITIES

DEFAULT_SEED = 42
DEFAULT_NUM_THREADS = 4


def get_deterministic_lgb_params(seed=DEFAULT_SEED, num_threads=DEFAULT_NUM_THREADS, **overrides):
    """Default LightGBM parameters with fixed seeds."""
    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": num_threads,
        "verbose": -1,
    }
    params.update(overrides)
    return params


def get_ridge_feature_cols(X_df):
    """Select irradiance predictors for the in-fold Ridge model."""
    suffixes = ["ghi_h0", "ghi_dec", "ghi_hm1", "ghi_hp1", "clearsky_h0", "clearsky_dec"]
    aggregates = ["all_ghi_mean", "all_clearsky_mean"]

    candidates = [f"{city}_{suffix}" for city in RAD_CITIES for suffix in suffixes]
    return [col for col in candidates + aggregates if col in X_df.columns]


def compute_oof_ridge_and_pred(X_tr_full, y_cf_tr, cap_tr, dt_tr, X_target_full, alpha=500.0, n_splits=6):
    """Out-of-fold Ridge predictions via Year-Month GroupKFold on training rows, plus full-fold Ridge predictions for target rows."""
    lin_cols = get_ridge_feature_cols(X_tr_full)
    A_tr = X_tr_full[lin_cols].values.astype(np.float64)
    A_tg = X_target_full[lin_cols].values.astype(np.float64)

    ym_groups = (dt_tr.dt.year * 100 + dt_tr.dt.month).values
    n_unique_ym = len(np.unique(ym_groups))
    k = min(n_splits, n_unique_ym)
    assert k >= 2, f"Need at least 2 year-month groups for GroupKFold, got {n_unique_ym}"

    gkf = GroupKFold(n_splits=k)
    oof_pred = np.zeros(len(A_tr), dtype=np.float64)
    with threadpool_limits(limits=1, user_api="blas"):
        for fit_i, oof_i in gkf.split(A_tr, y_cf_tr, groups=ym_groups):
            pipe_fold = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            pipe_fold.fit(A_tr[fit_i], y_cf_tr[fit_i], ridge__sample_weight=cap_tr[fit_i])
            oof_pred[oof_i] = np.clip(pipe_fold.predict(A_tr[oof_i]), 0.0, 0.95)

        # Full-train Ridge for target (internal ES val / eval year / test year)
        pipe_full = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        pipe_full.fit(A_tr, y_cf_tr, ridge__sample_weight=cap_tr)
        target_pred = np.clip(pipe_full.predict(A_tg), 0.0, 0.95)
    return oof_pred, target_pred, lin_cols


def _prepare_target_and_weights(mode, y_cf, cap, X_full, ridge_pred=None):
    if mode == "cf":
        return y_cf.astype(np.float64), cap.astype(np.float64), None
    elif mode == "residual":
        assert ridge_pred is not None
        return (y_cf - ridge_pred).astype(np.float64), cap.astype(np.float64), None
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _reconstruct_cf(mode, raw_pred, X_full, ridge_pred=None):
    if mode == "cf":
        return np.clip(raw_pred, 0.0, 1.0)
    elif mode == "residual":
        assert ridge_pred is not None
        return np.clip(ridge_pred + raw_pred, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def fit_predict_forward_lgb(
    X_tr_full,
    night_tr,
    df_tr,
    X_ev_full,
    night_ev,
    df_ev,
    feature_cols,
    mode="cf",
    add_phys_feat=False,
    daytime_only=True,
    es_val_frac=0.15,
    max_boost_round=3000,
    early_stopping_rounds=80,
    seed=DEFAULT_SEED,
    num_threads=DEFAULT_NUM_THREADS,
    lgb_overrides=None,
    ridge_alpha=500.0,
):
    """Fit two-stage Ridge and LightGBM model using an internal whole-day early-stopping split, refit on all daytime training rows for best_iteration rounds, and predict on df_ev."""
    params = get_deterministic_lgb_params(seed=seed, num_threads=num_threads, **(lgb_overrides or {}))
    need_ridge = add_phys_feat or (mode == "residual")

    if mode == "raw_mw":
        # Raw MW target mode for baseline B1
        is_es_val = split_internal_es_days(df_tr["dt"], val_frac=es_val_frac, seed=seed)
        es_tr_mask = ~is_es_val
        es_va_mask = is_es_val

        X_tr_sub = X_tr_full[feature_cols].iloc[es_tr_mask]
        y_tr_sub = df_tr["generation_mw"].values[es_tr_mask]
        X_va_sub = X_tr_full[feature_cols].iloc[es_va_mask]
        y_va_sub = df_tr["generation_mw"].values[es_va_mask]

        dtr = lgb.Dataset(X_tr_sub, label=y_tr_sub, free_raw_data=False)
        dva = lgb.Dataset(X_va_sub, label=y_va_sub, reference=dtr, free_raw_data=False)
        bst_es = lgb.train(
            params,
            dtr,
            num_boost_round=max_boost_round,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        best_iter = int(bst_es.best_iteration) if bst_es.best_iteration and bst_es.best_iteration > 0 else max_boost_round
        best_iter = max(best_iter, 50)

        dfull = lgb.Dataset(X_tr_full[feature_cols], label=df_tr["generation_mw"].values, free_raw_data=False)
        bst_full = lgb.train(params, dfull, num_boost_round=best_iter)
        pred_mw = np.clip(bst_full.predict(X_ev_full[feature_cols]), 0.0, None)
        return pred_mw, best_iter, params

    # Select training rows (daytime-only or all rows)
    tr_keep = (~night_tr) if daytime_only else np.ones(len(df_tr), dtype=bool)
    ev_keep = (~night_ev) if daytime_only else np.ones(len(df_ev), dtype=bool)

    tr_idx_keep = np.where(tr_keep)[0]
    ev_idx_keep = np.where(ev_keep)[0]

    # Step 1: Whole-day 15% internal early-stopping split within df_tr
    is_es_val_all = split_internal_es_days(df_tr["dt"], val_frac=es_val_frac, seed=seed)
    es_tr_idx = tr_idx_keep[~is_es_val_all[tr_idx_keep]]
    es_va_idx = tr_idx_keep[is_es_val_all[tr_idx_keep]]

    X_es_tr = X_tr_full.iloc[es_tr_idx][feature_cols].copy()
    X_es_va = X_tr_full.iloc[es_va_idx][feature_cols].copy()
    y_cf_es_tr = df_tr["cf"].values[es_tr_idx]
    y_cf_es_va = df_tr["cf"].values[es_va_idx]
    cap_es_tr = df_tr["installed_capacity_mwp"].values[es_tr_idx]
    cap_es_va = df_tr["installed_capacity_mwp"].values[es_va_idx]

    r_es_tr, r_es_va = None, None
    if need_ridge:
        r_es_tr, r_es_va, _ = compute_oof_ridge_and_pred(
            X_tr_full.iloc[es_tr_idx],
            y_cf_es_tr,
            cap_es_tr,
            df_tr["dt"].iloc[es_tr_idx],
            X_tr_full.iloc[es_va_idx],
            alpha=ridge_alpha,
        )
        if add_phys_feat:
            X_es_tr["phys_ridge_cf"] = r_es_tr.astype(np.float32)
            X_es_va["phys_ridge_cf"] = r_es_va.astype(np.float32)

    y_tr_t, w_tr_t, _ = _prepare_target_and_weights(mode, y_cf_es_tr, cap_es_tr, X_tr_full.iloc[es_tr_idx], r_es_tr)
    y_va_t, w_va_t, _ = _prepare_target_and_weights(mode, y_cf_es_va, cap_es_va, X_tr_full.iloc[es_va_idx], r_es_va)

    dtr_es = lgb.Dataset(X_es_tr, label=y_tr_t, weight=w_tr_t, free_raw_data=False)
    dva_es = lgb.Dataset(X_es_va, label=y_va_t, weight=w_va_t, reference=dtr_es, free_raw_data=False)
    bst_es = lgb.train(
        params,
        dtr_es,
        num_boost_round=max_boost_round,
        valid_sets=[dva_es],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    best_iter = int(bst_es.best_iteration) if bst_es.best_iteration and bst_es.best_iteration > 0 else max_boost_round
    best_iter = max(best_iter, 50)

    # Step 2: Refit on 100% of fold's training period for `best_iter` rounds
    X_full_tr = X_tr_full.iloc[tr_idx_keep][feature_cols].copy()
    X_full_ev = X_ev_full.iloc[ev_idx_keep][feature_cols].copy()
    y_cf_full_tr = df_tr["cf"].values[tr_idx_keep]
    cap_full_tr = df_tr["installed_capacity_mwp"].values[tr_idx_keep]
    cap_full_ev = df_ev["installed_capacity_mwp"].values[ev_idx_keep]

    r_full_tr, r_full_ev = None, None
    if need_ridge:
        r_full_tr, r_full_ev, _ = compute_oof_ridge_and_pred(
            X_tr_full.iloc[tr_idx_keep],
            y_cf_full_tr,
            cap_full_tr,
            df_tr["dt"].iloc[tr_idx_keep],
            X_ev_full.iloc[ev_idx_keep],
            alpha=ridge_alpha,
        )
        if add_phys_feat:
            X_full_tr["phys_ridge_cf"] = r_full_tr.astype(np.float32)
            X_full_ev["phys_ridge_cf"] = r_full_ev.astype(np.float32)

    y_full_t, w_full_t, _ = _prepare_target_and_weights(
        mode, y_cf_full_tr, cap_full_tr, X_tr_full.iloc[tr_idx_keep], r_full_tr
    )
    dfull = lgb.Dataset(X_full_tr, label=y_full_t, weight=w_full_t, free_raw_data=False)
    bst_full = lgb.train(params, dfull, num_boost_round=best_iter)

    # Step 3: Single-shot prediction on evaluation/test period
    raw_pred_ev = bst_full.predict(X_full_ev)
    pred_cf_ev = _reconstruct_cf(mode, raw_pred_ev, X_ev_full.iloc[ev_idx_keep], r_full_ev)

    pred_mw_full = np.zeros(len(df_ev), dtype=np.float64)
    pred_mw_full[ev_idx_keep] = pred_cf_ev * cap_full_ev
    return pred_mw_full, best_iter, params

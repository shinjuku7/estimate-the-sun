"""The four base learners. All predict capacity factor (MW / installed MWp) on sunlit half-hours.
run.py fits all four on each of the two feature views (8 models) and averages them.

With estimated locations and the held-level correction, the eight-model mean
scored 109.50 MW on July 2023-June 2024 and 112.92 MW on July 2024-June 2025.
"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

SEED = 2025
THREADS = 4

LGB_COMMON = dict(bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=THREADS, seed=SEED, deterministic=True, force_row_wise=True)

CONFIGS = {
    'lgb_l1_big': ('lgb', dict(LGB_COMMON, objective='l1', learning_rate=0.02, num_leaves=255,
                                feature_fraction=0.4, min_data_in_leaf=100), 3000),
    'lgb_l2': ('lgb', dict(LGB_COMMON, objective='l2', learning_rate=0.03, num_leaves=127,
                           feature_fraction=0.5, min_data_in_leaf=80), 3000),
    'xgb_l1': ('xgb', dict(n_estimators=3000, learning_rate=0.03, max_depth=8, subsample=0.8,
                           colsample_bytree=0.5, min_child_weight=20, objective='reg:absoluteerror',
                           tree_method='hist', n_jobs=THREADS, random_state=SEED), None),
    'cat_l1': ('cat', dict(iterations=4000, learning_rate=0.06, depth=8, loss_function='MAE',
                           random_seed=SEED, verbose=0, thread_count=THREADS), None),
}

# fast model used only to measure the level drift (leave-one-quarter-out)
DRIFT_MODEL = (dict(LGB_COMMON, objective='l1', learning_rate=0.06, num_leaves=63,
                    feature_fraction=0.7, min_data_in_leaf=40), 1500)


def fit(kind, params, rounds, X, y):
    if kind == 'lgb':
        return lgb.train(params, lgb.Dataset(X, y), num_boost_round=rounds)
    if kind == 'xgb':
        m = xgb.XGBRegressor(**params)
        m.fit(X, y)
        return m
    if kind == 'cat':
        m = CatBoostRegressor(**params)
        m.fit(X, y)
        return m
    raise ValueError(kind)


def predict(model, X):
    return np.clip(model.predict(X), 0, None)


def fit_predict_ensemble(Xtr, ytr, Xte, names=None, log=print):
    names = names or list(CONFIGS)
    preds = {}
    for n in names:
        kind, params, rounds = CONFIGS[n]
        m = fit(kind, params, rounds, Xtr, ytr)
        preds[n] = predict(m, Xte)
        log(f'  fitted {n}')
    return np.mean([preds[n] for n in names], axis=0), preds

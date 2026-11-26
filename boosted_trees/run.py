#!/usr/bin/env python3
"""solar-britain (Square1 Ai, "Estimate the sun").

  python run.py --data data --out-dir submissions   # fit on all of train.csv, predict test.csv
  python run.py --data data --validate              # the two forward one-year backtests

Writes two files from one fit (same models, different level for the test year):
  submissions/submission_persist.csv  level held at its end-of-training value (main submission)
  submissions/submission_linear.csv   level trend continued through the test year (hedge)

data/ must contain train.csv and test.csv exactly as downloaded from the competition page.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
warnings.filterwarnings('ignore')

import features            # noqa: E402
import drift               # noqa: E402
import models              # noqa: E402

DROP = {'id', 'dt', 'is_train', 'y', 'date', 'cap'}
FOLDS = {'F1': ('2024-07-01', '2025-07-01'), 'F2': ('2023-07-01', '2024-07-01')}
T_START = time.time()


def log(msg):
    print(f'[{time.time() - T_START:6.0f}s] {msg}', flush=True)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def fit_and_predict(views, trn_mask, tst_mask, names=None):
    """views: {name: (feature frame over all rows, feature list)}; the first view is the aligned one.
    trn_mask / tst_mask: boolean row masks. Returns uncorrected MW predictions for the test rows, the
    per-model predictions, and the two level-factor functions measured on the training window."""
    F0, feats0 = views['aligned']
    trn, tst = F0[trn_mask], F0[tst_mask]
    lit_tr = (trn.sun_any > 0).values          # sunlit rows, always judged on the aligned (correct) period
    lit = (tst.sun_any > 0).values
    trd, ted = trn[lit_tr], tst[lit]
    params, rounds = models.DRIFT_MODEL
    oof = drift.oof_quarters(trd, feats0, params, rounds, log=log)
    levels = {lv: drift.level_factor(trd, oof, level=lv) for lv in ('persist', 'linear')}
    log(f"drift: {json.dumps(levels['persist'][1], default=float)}")
    y = trd.y.values / trd.cap.values
    per = {}
    for vname, (Fv, fv) in views.items():
        Xtr = Fv[trn_mask][lit_tr][fv]
        Xte = Fv[tst_mask][lit][fv]
        _, pm = models.fit_predict_ensemble(Xtr, y, Xte, names=names, log=lambda m: log(f'{vname}: {m}'))
        for n, p in pm.items():
            a = np.zeros(len(tst)); a[lit] = p * ted.cap.values   # sun down everywhere -> exactly 0
            per[f'{vname}/{n}'] = a
    raw = np.mean(list(per.values()), axis=0)
    return raw, per, levels


def apply_level(raw, tst, levels, level):
    fn, info = levels[level]
    return raw * fn(tst.dt), info


def validate(views, save_preds=None, folds=None):
    F0 = views['aligned'][0]
    saved = []
    for fold, (vs, ve) in FOLDS.items():
        if folds and fold not in folds:
            continue
        vs, ve = pd.Timestamp(vs, tz='UTC'), pd.Timestamp(ve, tz='UTC')
        trn_mask = ((F0.is_train == 1) & (F0.dt < vs)).values
        val_mask = ((F0.is_train == 1) & (F0.dt >= vs) & (F0.dt < ve)).values
        val = F0[val_mask]
        log(f'{fold}: train {F0[trn_mask].dt.min():%Y-%m-%d}..{F0[trn_mask].dt.max():%Y-%m-%d}, '
            f'validate {val.dt.min():%Y-%m-%d}..{val.dt.max():%Y-%m-%d}')
        raw, per, levels = fit_and_predict(views, trn_mask, val_mask)
        y = val.y.values
        ss = np.arange(0.90, 1.05, 0.0025)
        oracle = min((mae(raw * s, y), s) for s in ss)
        res = {lv: apply_level(raw, val, levels, lv)[0] for lv in levels}
        log(f'{fold} RESULT  uncorrected {mae(raw, y):.2f}  persist {mae(res["persist"], y):.2f} '
            f'(bias {np.mean(res["persist"] - y):+.1f})  linear {mae(res["linear"], y):.2f} '
            f'(bias {np.mean(res["linear"] - y):+.1f})  oracle-scale {oracle[0]:.2f} at {oracle[1]:.4f}')
        f_persist = levels['persist'][0](val.dt)
        saved.append(pd.DataFrame({'fold': fold, 'id': val.id.values, 'datetime_utc': val.dt.values, 'y': y,
                                   'pred_uncorrected': raw, 'pred_persist': res['persist'],
                                   'pred_linear': res['linear']}))
        for vname in views:
            sub = [p for k, p in per.items() if k.startswith(vname + '/')]
            log(f'{fold}   view {vname:8s} (4 models) persist {mae(np.mean(sub, axis=0) * f_persist, y):.2f}')
        for n, p in per.items():
            log(f'{fold}   {n:20s} persist {mae(p * f_persist, y):.2f}')
    if save_preds:
        pd.concat(saved).to_csv(save_preds, index=False)
        log(f'wrote out-of-time predictions to {save_preds}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--out-dir', default='submissions')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--save-preds', default=None, help='with --validate: write per-row fold predictions here')
    ap.add_argument('--folds', nargs='*', default=None, help='with --validate: only these folds (F1 F2)')
    args = ap.parse_args()

    df = features.load(args.data)
    points = features.estimate_points(df)
    log(f'point locations estimated from the clear-sky column: {points}')
    views = {}
    for vname, label_is_end in (('aligned', True), ('datacard', False)):
        Fv = features.build(df, label_is_end=label_is_end, points=points)
        views[vname] = (Fv, [c for c in Fv.columns if c not in DROP])
    assert (views['aligned'][0].dt.values == views['datacard'][0].dt.values).all()
    F0 = views['aligned'][0]
    log(f'features built: {len(F0)} rows, {len(views["aligned"][1])} features per view')

    if args.validate:
        validate(views, args.save_preds, args.folds)
        return

    trn_mask = (F0.is_train == 1).values
    tst_mask = (F0.is_train == 0).values
    tst = F0[tst_mask]
    raw, per, levels = fit_and_predict(views, trn_mask, tst_mask)
    order = pd.read_csv(os.path.join(args.data, 'test.csv'), usecols=['id'])
    os.makedirs(args.out_dir, exist_ok=True)
    for level in ('persist', 'linear'):
        pred, info = apply_level(raw, tst, levels, level)
        sub = order.merge(pd.DataFrame({'id': tst.id.values, 'generation_mw': np.round(pred, 4)}),
                          on='id', how='left')
        assert sub.generation_mw.notna().all() and len(sub) == len(order)
        path = os.path.join(args.out_dir, f'submission_{level}.csv')
        sub.to_csv(path, index=False, lineterminator="\n")
        log(f'wrote {path}: mean {sub.generation_mw.mean():.1f} MW '
            f'(uncorrected {raw.mean():.1f}), {json.dumps(info, default=float)}')


if __name__ == '__main__':
    main()

"""Level drift correction.

Observation: at fixed weather, PV_Live generation per MWp of PV_Live capacity falls by roughly 1.5-2 %
a year in this data, and faster when the sun is low. A model trained on all years therefore predicts
the year after training too high (+1.3 % and +2.4 % mean bias on the two forward folds).

Procedure (uses training targets only):
  1. leave-one-quarter-out predictions over the training window with a fast LightGBM;
  2. monthly log(actual / predicted), weighted least squares against time;
  3. the factor applied to the next year is the fitted line evaluated at the END of training
     ("persistence of the latest level"). Extending the line into the test year over-corrected on
     both backtests, so it is not used.
Optionally the line is fitted separately for bands of solar elevation.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

T0 = pd.Timestamp('2022-01-01', tz='UTC')
ELEV_BANDS = [-10, 5, 10, 20, 35, 90]


def t_years(dt):
    return (dt - T0).dt.total_seconds().values / (365.25 * 86400)


def oof_quarters(trd, feats, params, rounds, log=print):
    """Leave-one-quarter-out predictions (MW) for sunlit training rows."""
    q = (trd.dt.dt.year.astype(str) + 'Q' + trd.dt.dt.quarter.astype(str)).values
    oof = np.zeros(len(trd))
    y = trd.y.values / trd.cap.values
    for qq in sorted(set(q)):
        a = q != qq
        m = lgb.train(params, lgb.Dataset(trd.loc[a, feats], y[a]), num_boost_round=rounds)
        oof[~a] = np.clip(m.predict(trd.loc[~a, feats]), 0, None) * trd.cap.values[~a]
        log(f'  drift OOF {qq} done')
    return oof


def _wls_line(t, r, w):
    A = np.vstack([np.ones(len(t)), t]).T
    sw = np.sqrt(w)
    return np.linalg.lstsq(A * sw[:, None], r * sw, rcond=None)[0]


def monthly_line(dt, y, p):
    d = pd.DataFrame({'t': t_years(dt), 'y': y, 'p': p, 'ym': dt.dt.strftime('%Y-%m').values})
    g = d.groupby('ym').agg(t=('t', 'mean'), y=('y', 'sum'), p=('p', 'sum'))
    g = g[(g.p > 0) & (g.y > 0)]
    return _wls_line(g.t.values, np.log(g.y / g.p).values, g.p.values), g


def level_factor(trd, oof, level='persist'):
    """Returns (function(dt) -> factor, info).

    level='persist': the fitted line evaluated at the last training timestamp, constant afterwards
                     (best on both backtests; this is the default and the main submission).
    level='linear' : the fitted line continued through the prediction period (over-corrected on both
                     backtests; kept only as the second, hedging submission).
    """
    t_end = t_years(trd.dt).max()
    coef, _ = monthly_line(trd.dt, trd.y.values, oof)
    g_fac = float(np.exp(coef[0] + coef[1] * t_end))
    info = {'slope_pct_per_year': 100 * coef[1], 'factor_at_train_end': g_fac, 'level': level}
    if level == 'persist':
        return (lambda dt: np.full(len(dt), g_fac)), info
    if level == 'linear':
        return (lambda dt: np.exp(coef[0] + coef[1] * np.maximum(t_years(dt), t_end))), info
    raise ValueError(level)


def persistence_factor(trd, oof, by_elevation=False):
    """Returns a function elev -> multiplicative factor for the year after training."""
    t_end = t_years(trd.dt).max()
    coef, _ = monthly_line(trd.dt, trd.y.values, oof)
    g_fac = float(np.exp(coef[0] + coef[1] * t_end))
    info = {'slope_pct_per_year': 100 * coef[1], 'global_factor': g_fac}
    if not by_elevation:
        return (lambda elev: np.full(len(elev), g_fac)), info
    band = pd.cut(trd.elev_c, ELEV_BANDS, labels=False).values
    facs = {}
    for b in range(len(ELEV_BANDS) - 1):
        m = band == b
        c, _ = monthly_line(trd.dt[m], trd.y.values[m], oof[m])
        facs[b] = float(np.exp(c[0] + c[1] * t_end))
    info['band_factors'] = facs

    def f(elev):
        bb = pd.cut(pd.Series(elev), ELEV_BANDS, labels=False).fillna(0).astype(int).values
        return np.array([facs[x] for x in bb])
    return f, info

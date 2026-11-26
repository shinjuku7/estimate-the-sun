"""Feature construction (one row per half-hour, train and test together so neighbouring hours are available).

Facts the features rely on (verified on train):
  * weather rows HH:00 and HH:30 are identical and describe hour [HH:00, HH+1:00)
  * target at label t is the half-hour [t-30, t)  (label = period end)
  * target == 0 exactly whenever the sun is down over the whole of GB during the period
"""
import numpy as np
import pandas as pd
from solar import POINT_NAMES, estimate_points, solar_position, mean_cosz

PTS = list(POINT_NAMES)
IRR_PTS = [p for p in PTS if p != 'cardiff']   # cardiff irradiance is bristol's grid cell
TILT = 35.0


def load(data_dir='data'):
    tr = pd.read_csv(f'{data_dir}/train.csv')
    te = pd.read_csv(f'{data_dir}/test.csv')
    df = pd.concat([tr.assign(is_train=1), te.assign(is_train=0)], ignore_index=True)
    df['dt'] = pd.to_datetime(df['datetime_utc'], utc=True)
    df = df.sort_values('dt').reset_index(drop=True)
    assert (df.dt.diff().dropna() == pd.Timedelta(minutes=30)).all()
    return df


def _poa_factor(start, lat, lon, tilt=TILT, step=3):
    """Mean over the period of cos(incidence) on a south-facing tilted plane, clipped at 0, and mean cos zenith."""
    offs = np.arange(step / 2, 30, step)
    inc = np.zeros(len(start)); cz_acc = np.zeros(len(start))
    for o in offs:
        el, az, cz, e0 = solar_position(start + pd.Timedelta(minutes=float(o)), lat, lon)
        zen = np.radians(90 - el); azr = np.radians(az)
        ci = np.cos(zen) * np.cos(np.radians(tilt)) + np.sin(zen) * np.sin(np.radians(tilt)) * np.cos(azr - np.pi)
        up = cz > 0
        inc += np.where(up, np.clip(ci, 0, None), 0) * e0
        cz_acc += np.clip(cz, 0, None) * e0
    return inc / len(offs), cz_acc / len(offs)


def build(df, label_is_end=True, points=None):
    """label_is_end=True : the target period is [t-30, t)  (what the data shows; the main view)
    label_is_end=False: the target period is [t, t+30) (the data card's wording). Wrong for the target,
                        but it interpolates the hourly weather towards the other neighbouring hour and
                        shifts the sun-position features, and models built on it make different errors.
                        Used only as a second view in the ensemble."""
    POINTS = points if points is not None else estimate_points(df)
    CENTROID = (float(np.mean([v[0] for v in POINTS.values()])), float(np.mean([v[1] for v in POINTS.values()])))
    c = {}
    t = df['dt']
    if label_is_end:
        centre = t - pd.Timedelta(minutes=15)
        start = pd.DatetimeIndex(t - pd.Timedelta(minutes=30))
        second_half = (t.dt.minute == 0).values
    else:
        centre = t + pd.Timedelta(minutes=15)
        start = pd.DatetimeIndex(t)
        second_half = (t.dt.minute == 30).values
    hourly = df.loc[df.dt.dt.minute == 0].set_index('dt')
    own = pd.DatetimeIndex(centre.dt.floor('h'))
    adj = own + pd.to_timedelta(np.where(second_half, 1, -1), unit='h')
    opp = own + pd.to_timedelta(np.where(second_half, -1, 1), unit='h')   # hour on the far side
    take = lambda col, hrs: hourly[col].reindex(hrs).values

    c['id'] = df['id'].values
    c['is_train'] = df['is_train'].values
    c['cap'] = df['installed_capacity_mwp'].values
    if 'generation_mw' in df:
        c['y'] = df['generation_mw'].values

    el, az, cz, e0 = solar_position(centre, *CENTROID)
    c['elev_c'] = el; c['az_c'] = az
    poa_c, sun_c = _poa_factor(start, *CENTROID)
    c['sun_c'] = sun_c; c['poa_c'] = poa_c
    c['doy'] = t.dt.dayofyear.values
    c['tod'] = (centre.dt.hour + centre.dt.minute / 60).values
    c['second_half'] = second_half.astype(int)
    sun_any = sun_c.copy()
    for p in ['norwich', 'plymouth', 'glasgow', 'london', 'edinburgh', 'southampton']:
        sun_any = np.maximum(sun_any, mean_cosz(start, 30, *POINTS[p], step_min=3))
    c['sun_any'] = sun_any

    G, C, K, PO = [], [], [], []
    for p in PTS:
        lat, lon = POINTS[p]
        if p in IRR_PTS:
            ghi_o, cs_o = take(f'{p}_ghi', own), take(f'{p}_clearsky', own)
            ghi_a, cs_a = take(f'{p}_ghi', adj), take(f'{p}_clearsky', adj)
            ghi_f, cs_f = take(f'{p}_ghi', opp), take(f'{p}_clearsky', opp)
            poa_h, m_half = _poa_factor(start, lat, lon)
            m_hour = mean_cosz(own, 60, lat, lon, step_min=3)
            ratio = np.where(m_hour > 1e-4, m_half / np.maximum(m_hour, 1e-4), 0.0)
            cs_half = cs_o * ratio
            kt = lambda g, cs: np.where(cs > 5, g / np.maximum(cs, 1e-6), np.nan)
            kt_o, kt_a, kt_f = kt(ghi_o, cs_o), kt(ghi_a, cs_a), kt(ghi_f, cs_f)
            kt_i = np.where(np.isnan(kt_a), kt_o, 0.75 * kt_o + 0.25 * kt_a)
            ghi_half = np.nan_to_num(kt_i * cs_half, nan=0.0)
            # crude plane-of-array proxy: beam-ish part scales with tilted/horizontal geometry ratio
            geo = np.where(m_half > 1e-3, poa_h / np.maximum(m_half, 1e-3), 0.0)
            geo = np.clip(geo, 0, 4)
            ktc = np.clip(np.nan_to_num(kt_i, nan=0.0), 0, 1.2)
            poa = ghi_half * ((1 - ktc) * 1.0 + ktc * geo)  # overcast ~ isotropic, clear ~ geometric
            c[f'{p}_ghi_o'] = ghi_o; c[f'{p}_ghi_a'] = ghi_a
            c[f'{p}_cs_half'] = cs_half
            c[f'{p}_kt_o'] = kt_o; c[f'{p}_kt_a'] = kt_a; c[f'{p}_kt_f'] = kt_f; c[f'{p}_kt_i'] = kt_i
            c[f'{p}_ghi_half'] = ghi_half; c[f'{p}_poa'] = poa
            G.append(ghi_half); C.append(cs_half); K.append(kt_i); PO.append(poa)
        for v in ['temp_c', 'cloud_pct', 'wind_ms', 'rh_pct']:
            vo, va = take(f'{p}_{v}', own), take(f'{p}_{v}', adj)
            c[f'{p}_{v}'] = 0.75 * vo + 0.25 * va
    G, C, K, PO = map(np.vstack, (G, C, K, PO))
    c['ghi_half_mean'] = G.mean(0); c['cs_half_mean'] = C.mean(0); c['poa_mean'] = PO.mean(0)
    with np.errstate(all='ignore'):
        c['kt_mean'] = np.nanmean(K, 0); c['kt_std'] = np.nanstd(K, 0)
        c['kt_min'] = np.nanmin(K, 0); c['kt_max'] = np.nanmax(K, 0)
    c['temp_mean'] = np.mean([c[f'{p}_temp_c'] for p in PTS], 0)
    c['cloud_mean'] = np.mean([c[f'{p}_cloud_pct'] for p in PTS], 0)
    c['wind_mean'] = np.mean([c[f'{p}_wind_ms'] for p in PTS], 0)
    c['rh_mean'] = np.mean([c[f'{p}_rh_pct'] for p in PTS], 0)
    out = pd.DataFrame(c)
    out['dt'] = df['dt'].reset_index(drop=True)
    # temporal context of national aggregates (30-min steps)
    extra = {}
    for k in (-4, -3, -2, -1, 1, 2, 3, 4):
        extra[f'ghi_half_mean_l{k}'] = out['ghi_half_mean'].shift(k)
        extra[f'kt_mean_l{k}'] = out['kt_mean'].shift(k)
    date = centre.dt.date.values
    tmp = pd.DataFrame({'date': date, 'g': out['ghi_half_mean'], 'cs': out['cs_half_mean']})
    extra['day_kt'] = tmp.groupby('date')['g'].transform('sum') / tmp.groupby('date')['cs'].transform('sum').replace(0, np.nan)
    out = pd.concat([out, pd.DataFrame(extra)], axis=1)
    out['date'] = date
    return out

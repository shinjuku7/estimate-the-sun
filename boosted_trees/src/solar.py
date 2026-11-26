"""NOAA solar position (vectorised), and point locations estimated from the shipped data."""
import numpy as np
import pandas as pd

POINT_NAMES = ['london', 'birmingham', 'manchester', 'leeds', 'bristol', 'norwich',
               'southampton', 'plymouth', 'cardiff', 'newcastle', 'glasgow', 'edinburgh']


def estimate_points(df, n_sample=4000, seed=0):
    """Effective (lat, lon) of each weather point, estimated from the shipped clear-sky column only.

    No external coordinates are used. For each point, a grid search finds the location whose
    computed sun geometry (Haurwitz-shaped clear-sky curve, scaled by least squares) best matches
    that point's hourly clear-sky irradiance. The result is an effective location for the data's
    grid cell, not the city itself (it sits 0.5-1 degree south of the cities, which does not matter
    because only consistent sun geometry is needed).
    """
    h = df[df.dt.dt.minute == 0].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(h), min(n_sample, len(h)), replace=False))
    ts = pd.DatetimeIndex(h.dt.values[idx])
    if ts.tz is None:
        ts = ts.tz_localize('UTC')

    def shape(lat, lon, step=10):
        offs = np.arange(step / 2, 60, step)
        acc = np.zeros(len(ts))
        for o in offs:
            _, _, cz, e0 = solar_position(ts + pd.Timedelta(minutes=float(o)), lat, lon)
            cz = np.clip(cz, 1e-4, None)
            acc += np.where(cz > 1e-3, cz * np.exp(-0.057 / cz), 0) * e0
        return acc / len(offs)

    pts, cache = {}, {}
    for p in POINT_NAMES:
        y = h[f'{p}_clearsky'].values[idx]
        key = y.tobytes()
        if key in cache:                      # identical clear-sky column (bristol/cardiff): same cell
            pts[p] = cache[key]
            continue

        def loss(lat, lon):
            s = shape(lat, lon)
            a = (s @ y) / (s @ s)
            return float(np.mean(np.abs(y - a * s)))
        best = min((loss(la, lo), la, lo) for la in np.arange(49, 60.01, 0.5) for lo in np.arange(-7, 3.01, 0.5))
        for step in (0.1, 0.02):
            _, la0, lo0 = best
            best = min([best] + [(loss(la, lo), la, lo)
                                 for la in np.arange(la0 - 5 * step, la0 + 5.01 * step, step)
                                 for lo in np.arange(lo0 - 5 * step, lo0 + 5.01 * step, step)])
        pts[p] = cache[key] = (round(float(best[1]), 3), round(float(best[2]), 3))
    return pts

def solar_position(ts_utc, lat, lon):
    """ts_utc: pandas DatetimeIndex/Series (tz-aware UTC). Returns (elevation_deg, azimuth_deg, cos_zenith)."""
    ts = pd.DatetimeIndex(ts_utc)
    jd = ts.to_julian_date().values
    jc = (jd - 2451545.0) / 36525.0
    gml = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    gma = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    gma_r = np.radians(gma)
    eqc = (np.sin(gma_r) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
           + np.sin(2 * gma_r) * (0.019993 - 0.000101 * jc) + np.sin(3 * gma_r) * 0.000289)
    true_long = gml + eqc
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))
    mean_obl = 23 + (26 + ((21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813)))) / 60) / 60
    obl_corr = mean_obl + 0.00256 * np.cos(np.radians(omega))
    decl = np.degrees(np.arcsin(np.sin(np.radians(obl_corr)) * np.sin(np.radians(app_long))))
    y = np.tan(np.radians(obl_corr / 2)) ** 2
    gml_r = np.radians(gml)
    eqt = 4 * np.degrees(y * np.sin(2 * gml_r) - 2 * ecc * np.sin(gma_r)
                         + 4 * ecc * y * np.sin(gma_r) * np.cos(2 * gml_r)
                         - 0.5 * y * y * np.sin(4 * gml_r) - 1.25 * ecc * ecc * np.sin(2 * gma_r))
    minutes = ts.hour.values * 60 + ts.minute.values + ts.second.values / 60.0
    tst = np.mod(minutes + eqt + 4 * lon, 1440.0)
    ha = tst / 4.0 - 180.0
    lat_r, decl_r, ha_r = np.radians(lat), np.radians(decl), np.radians(ha)
    cosz = np.sin(lat_r) * np.sin(decl_r) + np.cos(lat_r) * np.cos(decl_r) * np.cos(ha_r)
    cosz = np.clip(cosz, -1, 1)
    zen = np.degrees(np.arccos(cosz))
    elev = 90 - zen
    az = np.degrees(np.arctan2(np.sin(ha_r), np.cos(ha_r) * np.sin(lat_r) - np.tan(decl_r) * np.cos(lat_r))) + 180.0
    # extraterrestrial normal irradiance factor (earth-sun distance)
    doy = ts.dayofyear.values
    e0 = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    return elev, np.mod(az, 360.0), cosz, e0

def mean_cosz(ts_start, minutes_span, lat, lon, step_min=5):
    """Average of max(cos zenith,0) * E0 over [ts_start, ts_start+span) sampled every step_min minutes."""
    ts_start = pd.DatetimeIndex(ts_start)
    offs = np.arange(step_min / 2, minutes_span, step_min)
    acc = np.zeros(len(ts_start))
    for o in offs:
        _, _, cz, e0 = solar_position(ts_start + pd.Timedelta(minutes=float(o)), lat, lon)
        acc += np.clip(cz, 0, None) * e0
    return acc / len(offs)

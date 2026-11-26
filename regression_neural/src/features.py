import numpy as np
import pandas as pd
from utils import CITIES, VARS

# 11 unique radiation/cloud cities (cardiff shares identical GHI/clearsky/cloud grid with bristol)
RAD_CITIES = [c for c in CITIES if c != "cardiff"]


def build_raw_features(df_chron):
    """Raw shipped weather + calendar (+ installed_capacity_mwp) for baseline comparisons."""
    weather_cols = [f"{c}_{v}" for c in CITIES for v in VARS]
    X = df_chron[weather_cols + ["installed_capacity_mwp"]].copy()
    dt = df_chron["dt"]
    X["month"] = dt.dt.month
    X["doy"] = dt.dt.day_of_year
    X["hour"] = dt.dt.hour
    X["minute"] = dt.dt.minute
    X["hh"] = dt.dt.hour * 2 + dt.dt.minute // 30
    return X.astype(np.float32)


def build_feature_table(df_chron, return_groups=False, alignment="end"):
    """Build spatiotemporal weather and calendar features from a contiguous 30-minute DataFrame."""
    dt = df_chron["dt"]
    weather_cols = [f"{c}_{v}" for c in CITIES for v in VARS]

    hr = df_chron[dt.dt.minute == 0][["dt"] + weather_cols].copy().sort_values("dt").reset_index(drop=True)
    hr_idx = hr.set_index("dt")

    if alignment not in ("start", "end"):
        raise ValueError("alignment must be start or end")
    period_start = dt if alignment == "start" else dt - pd.Timedelta(minutes=30)
    true_hour = period_start.dt.floor("h")
    is_second_half = (period_start.dt.minute == 30).astype(np.int8).values
    sign_half = np.where(is_second_half == 1, 1.0, -1.0)

    shifts = {}
    for k in [-3, -2, -1, 0, 1, 2, 3]:
        shifts[k] = hr_idx.reindex(true_hour + pd.Timedelta(hours=k)).bfill().ffill()

    # Daily and rolling hourly weather statistics
    hr_date = hr["dt"].dt.date
    daily_ghi_sum = hr.groupby(hr_date)[[f"{c}_ghi" for c in RAD_CITIES]].transform("sum").mean(axis=1)
    daily_cs_sum = hr.groupby(hr_date)[[f"{c}_clearsky" for c in RAD_CITIES]].transform("sum").mean(axis=1)
    daily_cloud_mean = hr.groupby(hr_date)[[f"{c}_cloud_pct" for c in RAD_CITIES]].transform("mean").mean(axis=1)
    daily_cloud_std = hr.groupby(hr_date)[[f"{c}_cloud_pct" for c in RAD_CITIES]].transform("std").fillna(0.0).mean(axis=1)
    daily_temp_max = hr.groupby(hr_date)[[f"{c}_temp_c" for c in CITIES]].transform("max").mean(axis=1)
    daily_temp_min = hr.groupby(hr_date)[[f"{c}_temp_c" for c in CITIES]].transform("min").mean(axis=1)

    mean_temp_hr = hr[[f"{c}_temp_c" for c in CITIES]].mean(axis=1)
    mean_rh_hr = hr[[f"{c}_rh_pct" for c in CITIES]].mean(axis=1)
    mean_cloud_hr = hr[[f"{c}_cloud_pct" for c in RAD_CITIES]].mean(axis=1)
    hr_aux = pd.DataFrame({
        "daily_ghi_sum": daily_ghi_sum.values,
        "daily_cs_sum": daily_cs_sum.values,
        "daily_kt": daily_ghi_sum.values / np.maximum(daily_cs_sum.values, 1.0),
        "daily_cloud_mean": daily_cloud_mean.values,
        "daily_cloud_std": daily_cloud_std.values,
        "daily_temp_max": daily_temp_max.values,
        "daily_temp_min": daily_temp_min.values,
        "daily_temp_range": daily_temp_max.values - daily_temp_min.values,
        "temp_min_past12h": mean_temp_hr.rolling(12, min_periods=1).min().values,
        "temp_min_past24h": mean_temp_hr.rolling(24, min_periods=1).min().values,
        "temp_min_past48h": mean_temp_hr.rolling(48, min_periods=1).min().values,
        "temp_mean_past24h": mean_temp_hr.rolling(24, min_periods=1).mean().values,
        "hrs_below_1c_past24h": (mean_temp_hr < 1.0).rolling(24, min_periods=1).sum().values,
        "hrs_below_0c_past48h": (mean_temp_hr < 0.0).rolling(48, min_periods=1).sum().values,
        "rh_mean_past12h": mean_rh_hr.rolling(12, min_periods=1).mean().values,
        "cloud_mean_past12h": mean_cloud_hr.rolling(12, min_periods=1).mean().values,
    }, index=hr["dt"])
    aux_aligned = hr_aux.reindex(true_hour).bfill().ffill()

    g_base = {}
    g_temporal = {}
    g_spatial = {}
    g_weather_regime = {}

    # --- GROUP 1: Calendar & Base Aligned Weather ---
    doy = dt.dt.day_of_year.values.astype(np.float64)
    hh = (dt.dt.hour * 2 + dt.dt.minute // 30).values.astype(np.float64)
    g_base["doy"] = doy
    g_base["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    g_base["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    g_base["doy_sin2"] = np.sin(4.0 * np.pi * doy / 365.25)
    g_base["doy_cos2"] = np.cos(4.0 * np.pi * doy / 365.25)
    g_base["hh"] = hh
    g_base["hh_sin"] = np.sin(2.0 * np.pi * hh / 48.0)
    g_base["hh_cos"] = np.cos(2.0 * np.pi * hh / 48.0)
    g_base["is_second_half"] = is_second_half.astype(np.float32)

    interp_cache = {v: {} for v in VARS}
    h0_cache = {v: {} for v in VARS}
    hm1_cache = {v: {} for v in VARS}
    hp1_cache = {v: {} for v in VARS}
    hm2_cache = {v: {} for v in VARS}
    hp2_cache = {v: {} for v in VARS}

    for c in CITIES:
        for v in VARS:
            col = f"{c}_{v}"
            v0 = shifts[0][col].values.astype(np.float64)
            vm1 = shifts[-1][col].values.astype(np.float64)
            vp1 = shifts[1][col].values.astype(np.float64)
            vm2 = shifts[-2][col].values.astype(np.float64)
            vp2 = shifts[2][col].values.astype(np.float64)

            h0_cache[v][c] = v0
            hm1_cache[v][c] = vm1
            hp1_cache[v][c] = vp1
            hm2_cache[v][c] = vm2
            hp2_cache[v][c] = vp2

            # Area-preserving quadratic half-hour deconvolution
            v_dec = v0 + 0.125 * sign_half * (vp1 - vm1)
            if v in ("ghi", "clearsky", "wind_ms"):
                v_dec = np.maximum(v_dec, 0.0)
            elif v in ("cloud_pct", "rh_pct"):
                v_dec = np.clip(v_dec, 0.0, 100.0)
            interp_cache[v][c] = v_dec

            if c == "cardiff" and v in ("ghi", "clearsky", "cloud_pct"):
                continue

            g_base[f"{col}_h0"] = v0
            g_base[f"{col}_dec"] = v_dec

    # --- GROUP 2: Multi-Hour Temporal Context (H-3..H+3 + diffs + rolling) ---
    for c in RAD_CITIES:
        g_m1, g_0, g_p1 = hm1_cache["ghi"][c], h0_cache["ghi"][c], hp1_cache["ghi"][c]
        g_m2, g_p2 = hm2_cache["ghi"][c], hp2_cache["ghi"][c]
        cs_m1, cs_0, cs_p1 = hm1_cache["clearsky"][c], h0_cache["clearsky"][c], hp1_cache["clearsky"][c]
        cl_m1, cl_0, cl_p1 = hm1_cache["cloud_pct"][c], h0_cache["cloud_pct"][c], hp1_cache["cloud_pct"][c]

        kt_m1 = np.where(cs_m1 > 5.0, np.clip(g_m1 / np.maximum(cs_m1, 5.0), 0, 1.15), 0.0)
        kt_0 = np.where(cs_0 > 5.0, np.clip(g_0 / np.maximum(cs_0, 5.0), 0, 1.15), 0.0)
        kt_p1 = np.where(cs_p1 > 5.0, np.clip(g_p1 / np.maximum(cs_p1, 5.0), 0, 1.15), 0.0)

        g_temporal[f"{c}_ghi_hm1"] = g_m1
        g_temporal[f"{c}_ghi_hp1"] = g_p1
        g_temporal[f"{c}_ghi_hm2"] = g_m2
        g_temporal[f"{c}_ghi_hp2"] = g_p2
        g_temporal[f"{c}_kt_hm1"] = kt_m1
        g_temporal[f"{c}_kt_hp1"] = kt_p1
        g_temporal[f"{c}_kt_diff"] = kt_p1 - kt_m1
        g_temporal[f"{c}_kt_curv"] = (kt_p1 + kt_m1) * 0.5 - kt_0
        g_temporal[f"{c}_cloud_hm1"] = cl_m1
        g_temporal[f"{c}_cloud_hp1"] = cl_p1
        g_temporal[f"{c}_cloud_diff"] = cl_p1 - cl_m1

    for k in [-3, -2, -1, 0, 1, 2, 3]:
        g_mat = np.column_stack([shifts[k][f"{c}_ghi"].values for c in RAD_CITIES])
        cs_mat = np.column_stack([shifts[k][f"{c}_clearsky"].values for c in RAD_CITIES])
        cl_mat = np.column_stack([shifts[k][f"{c}_cloud_pct"].values for c in RAD_CITIES])
        t_mat = np.column_stack([shifts[k][f"{c}_temp_c"].values for c in CITIES])
        g_mean = g_mat.mean(axis=1)
        cs_mean = cs_mat.mean(axis=1)
        kt_mean = np.where(cs_mean > 5.0, g_mean / np.maximum(cs_mean, 5.0), 0.0)
        g_temporal[f"nat_ghi_shift{k:+d}"] = g_mean
        g_temporal[f"nat_kt_shift{k:+d}"] = np.clip(kt_mean, 0.0, 1.15)
        g_temporal[f"nat_cloud_shift{k:+d}"] = cl_mat.mean(axis=1)
        if k in (-2, -1, 1, 2):
            g_temporal[f"nat_temp_shift{k:+d}"] = t_mat.mean(axis=1)

    kt_stack3 = np.column_stack([g_temporal[f"nat_kt_shift{k:+d}"] for k in (-1, 0, 1)])
    kt_stack5 = np.column_stack([g_temporal[f"nat_kt_shift{k:+d}"] for k in (-2, -1, 0, 1, 2)])
    cl_stack5 = np.column_stack([g_temporal[f"nat_cloud_shift{k:+d}"] for k in (-2, -1, 0, 1, 2)])
    g_temporal["nat_kt_roll3_mean"] = kt_stack3.mean(axis=1)
    g_temporal["nat_kt_roll3_std"] = kt_stack3.std(axis=1)
    g_temporal["nat_kt_roll5_mean"] = kt_stack5.mean(axis=1)
    g_temporal["nat_kt_roll5_std"] = kt_stack5.std(axis=1)
    g_temporal["nat_cloud_roll5_mean"] = cl_stack5.mean(axis=1)
    g_temporal["nat_cloud_roll5_std"] = cl_stack5.std(axis=1)

    # --- GROUP 3: Weather aggregates across the named points ---
    cloud_mat = np.column_stack([interp_cache["cloud_pct"][c] for c in RAD_CITIES])
    temp_mat = np.column_stack([interp_cache["temp_c"][c] for c in CITIES])
    wind_mat = np.column_stack([interp_cache["wind_ms"][c] for c in CITIES])
    rh_mat = np.column_stack([interp_cache["rh_pct"][c] for c in CITIES])
    weather_matrices = [
        ("cloud", cloud_mat),
        ("temp", temp_mat),
        ("wind", wind_mat),
        ("rh", rh_mat),
    ]
    for variable in ("ghi", "clearsky"):
        values = np.column_stack([interp_cache[variable][city] for city in RAD_CITIES])
        weather_matrices.append((variable, values))

    for name, mat in weather_matrices:
        g_spatial[f"all_{name}_mean"] = mat.mean(axis=1)
        g_spatial[f"all_{name}_std"] = mat.std(axis=1)
        g_spatial[f"all_{name}_min"] = mat.min(axis=1)
        g_spatial[f"all_{name}_max"] = mat.max(axis=1)

    # --- GROUP 4: Daily Weather Regime & Cold/Snow/Frost Persistence ---
    for col in aux_aligned.columns:
        g_weather_regime[col] = aux_aligned[col].values

    max_cs_win = np.maximum(
        np.column_stack([h0_cache["clearsky"][c] for c in RAD_CITIES]).max(axis=1),
        np.maximum(
            np.column_stack([hm1_cache["clearsky"][c] for c in RAD_CITIES]).max(axis=1),
            np.column_stack([hp1_cache["clearsky"][c] for c in RAD_CITIES]).max(axis=1),
        ),
    )

    groups = {
        "base": list(g_base.keys()),
        "temporal": list(g_temporal.keys()),
        "spatial": list(g_spatial.keys()),
        "regime": list(g_weather_regime.keys()),
    }

    all_dict = {}
    for d in (g_base, g_temporal, g_spatial, g_weather_regime):
        for k, v in d.items():
            all_dict[k] = np.asarray(v, dtype=np.float32)

    X = pd.DataFrame(all_dict, index=df_chron.index)
    is_night = (max_cs_win == 0.0)

    if return_groups:
        return X, is_night, groups
    return X, is_night


def build_fold_features(df_train_part, df_eval_part, return_groups=False, alignment="end"):
    """Build features separately for train and evaluation splits, prepending 48 hours of prior weather to the evaluation split for rolling window continuity."""
    X_tr, night_tr, groups = build_feature_table(df_train_part.reset_index(drop=True), return_groups=True, alignment=alignment)

    common_cols = [c for c in df_eval_part.columns if c in df_train_part.columns and c not in ("generation_mw", "cf")]
    prefix_len = min(96, len(df_train_part))
    eval_ext = pd.concat(
        [df_train_part[common_cols].tail(prefix_len), df_eval_part[common_cols]],
        ignore_index=True,
    )
    X_ev_ext, night_ev_ext = build_feature_table(eval_ext, alignment=alignment)
    X_ev = X_ev_ext.iloc[prefix_len:].reset_index(drop=True)
    night_ev = night_ev_ext[prefix_len:]

    if return_groups:
        return X_tr, night_tr, X_ev, night_ev, groups
    return X_tr, night_tr, X_ev, night_ev

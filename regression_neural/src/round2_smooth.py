"""Smooth capacity-factor regression on local irradiance, clock and fleet size."""
import numpy as np
from scipy.linalg import solve
from scipy.ndimage import uniform_filter1d
from threadpoolctl import threadpool_limits

from features import RAD_CITIES


def columns(x, suffix):
    return x[[f"{c}_{suffix}" for c in RAD_CITIES]].to_numpy(dtype=float)


def row_interactions(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


def design(x):
    # All terms have fixed physical scales. Regularization shrinks interactions
    # more strongly than main effects. No quantiles/knots are fitted on future rows.
    g = columns(x, "ghi_dec") / 1000
    cs = columns(x, "clearsky_dec") / 1000
    gh = columns(x, "ghi_h0") / 1000
    ch = columns(x, "clearsky_h0") / 1000
    k = np.clip((g + 0.012) / (cs + 0.024), 0, 1.15)
    cloud = columns(x, "cloud_pct_dec") / 100
    temp = (columns(x, "temp_c_dec") - 12) / 15
    wind = columns(x, "wind_ms_dec") / 6
    rh = (columns(x, "rh_pct_dec") - 75) / 25
    # Astronomical clock terms from UTC only. No geographic lookup.
    phase = 2 * np.pi * (x.doy.to_numpy(dtype=float) - 1) / 365.25
    eq_min = 229.18 * (0.000075 + 0.001868*np.cos(phase) - 0.032077*np.sin(phase)
                       - 0.014615*np.cos(2*phase) - 0.040849*np.sin(2*phase))
    h = (x.hh.to_numpy(dtype=float)/2 - 0.25 + eq_min/60 - 12) * np.pi/12
    s = np.cos(phase - 2*np.pi*171/365.25)
    hour = np.stack([np.ones(len(x)), np.cos(h), np.sin(h), np.cos(2*h), np.sin(2*h)], axis=1)
    geometry = np.column_stack([hour, hour*s[:, None], hour*(2*s*s-1)[:, None],
                                np.cos(3*h), np.sin(3*h), np.sin(phase), np.cos(phase)])
    # Approximate direct and diffuse components, including low-sun response.
    beam = np.sqrt(cs) * k**2
    diffuse = g * (1-k)
    gm = g.mean(axis=1)
    cm = cs.mean(axis=1)
    km = k.mean(axis=1)
    site = [g, cs, beam, diffuse, g*k*k, g*temp, g*wind, g*rh,
            cs*cloud, g*cloud, gh-g, (ch-cs), cs*k*(1-k)]
    blocks = [*site, row_interactions(g, geometry)*0.6,
              row_interactions(beam, geometry)*0.45,
              row_interactions(diffuse, geometry)*0.45]
    # Spatially pooled meteorological responses receive more degrees of freedom.
    pool = np.column_stack([gm, cm, beam.mean(1), diffuse.mean(1), gm*km**2,
                            np.sqrt(cm), g.std(1), (g*temp).mean(1),
                            (cs*cloud).mean(1), (g*wind).mean(1), g.max(1)])
    blocks += [row_interactions(pool, geometry), geometry*0.15]
    for lag in ("hm1", "hp1", "hm2", "hp2"):
        gl = columns(x, "ghi_" + lag) / 1000
        blocks += [gl*0.3, row_interactions(gl.mean(1)[:, None], geometry)*0.4]
    for lag in ("hm1", "hp1"):
        kl = columns(x, "kt_" + lag)
        blocks += [cs*(kl-k), row_interactions((cs*(kl-k)).mean(1)[:, None], geometry)]
    for v, scale in (("daily_kt", 1), ("daily_temp_max", 20), ("temp_min_past24h", 15),
                     ("daily_cloud_std", 25), ("daily_temp_range", 10)):
        blocks.append(row_interactions((gm*x[v].to_numpy()/scale)[:, None], geometry)*0.5)
    return np.ascontiguousarray(np.column_stack(blocks), dtype=np.float64)


def weather_context(x):
    """Half-hour, multi-hour and whole-day transmission at each supplied point."""
    ghi = columns(x, "ghi_dec").reshape(-1,48,len(RAD_CITIES))/1000
    sky = columns(x, "clearsky_dec").reshape(ghi.shape)/1000
    local = ghi/np.maximum(sky,0.025)
    blocks = []
    for width in (5,13,48):
        if width == 48:
            transmission = ghi.sum(1,keepdims=True)/np.maximum(sky.sum(1,keepdims=True),0.025)
        else:
            g = uniform_filter1d(ghi,width,axis=1,mode="nearest")
            c = uniform_filter1d(sky,width,axis=1,mode="nearest")
            transmission = g/np.maximum(c,0.025)
        shift = (transmission-local)*sky
        blocks.extend([shift, shift*transmission, shift*np.sqrt(sky),
                       shift*columns(x,"wind_ms_dec").reshape(ghi.shape)/6])
    return np.column_stack([block.reshape(len(x),-1) for block in blocks])


def robust_ridge(a, target, base_weight, penalty=0.09):
    weight = base_weight
    for _ in range(4):
        weighted = a.T * weight
        gram = weighted @ a
        gram.flat[::len(gram)+1] += penalty
        beta = solve(gram, weighted @ target, assume_a="pos")
        weight = base_weight / np.maximum(1, np.abs(a @ beta-target)/0.009)
    return beta


def fleet_power(x, night, past):
    """Partial yield elasticity controlling for weather, shrunken toward constant CF."""
    if (past.dt.max()-past.dt.min()).days < 400:
        return 1.0
    a = design(x.loc[~night])
    cap = past.installed_capacity_mwp.to_numpy()[~night]
    y = past.cf.to_numpy()[~night]
    w = cap/cap.mean()
    with threadpool_limits(limits=2, user_api="blas"):
        proxy = np.clip(a @ robust_ridge(a, y, w), 0, 1)
        trend = proxy * np.log(cap / np.median(cap))
        coef = robust_ridge(np.column_stack([a, trend]), y, w)[-1]
    return float(np.clip(1 + coef, 0.55, 1.2))


def weather_basis(x):
    """Fixed Fourier features approximate smooth nonlinear weather interactions."""
    rng = np.random.default_rng(2026)
    g = columns(x, "ghi_dec")/1000
    cs = columns(x, "clearsky_dec")/1000
    k = np.clip((g+0.01)/(cs+0.02), 0, 1.2)
    temp = (columns(x, "temp_c_dec")-12)/10
    wind = (columns(x, "wind_ms_dec")-4)/3
    rh = (columns(x, "rh_pct_dec")-75)/20
    cloud = columns(x, "cloud_pct_dec")/100
    phase = 2*np.pi*(x.doy.to_numpy()-172)/365.25
    hour = np.pi*(x.hh.to_numpy()/2-12.25)/12
    season, clock = np.cos(phase), np.sin(hour)
    prior, after = columns(x, "kt_hm1"), columns(x, "kt_hp1")
    def mapping(count, width):
        return rng.normal(size=(count,width)), rng.uniform(0, 2*np.pi, width)
    w, offset = mapping(10, 40)
    blocks = []
    for i in range(k.shape[1]):
        local = np.column_stack([2*k[:,i], 2*cs[:,i], 1.5*cloud[:,i], temp[:,i]*0.6,
                                 rh[:,i]*0.4, wind[:,i]*0.4, season, clock,
                                 0.8*(prior[:,i]-k[:,i]), 0.8*(after[:,i]-k[:,i])])
        blocks.append(np.cos(local @ w + offset)*np.sqrt(cs[:,i,None])*0.12)
    envelope = 0.02+np.sqrt(cs.mean(1))
    common = [2*k.mean(1), 2*cs.mean(1), 1.5*cloud.mean(1), temp.mean(1)*0.7,
              wind.mean(1)*0.5, rh.mean(1)*0.5, season, clock, np.cos(hour),
              1.5*k.std(1), x.daily_kt.to_numpy(), (prior-after).mean(1)]
    w, offset = mapping(len(common), 192)
    blocks.append(np.cos(np.column_stack(common) @ w+offset)*envelope[:,None]*0.16)
    return np.column_stack(blocks)


def fit_smooth(x, night, past, xe, ne, future, fleet=False, nonlinear=False,
               recent=False, geometry=False, context=False, penalty=0.09, fleet_penalty=None,
               difference_weight=0.0, regional_fleet=False):
    a = design(x.loc[~night])
    b = design(xe.loc[~ne])
    cap = past.installed_capacity_mwp.to_numpy()[~night]
    cap_ev = future.installed_capacity_mwp.to_numpy()[~ne]
    target = past.cf.to_numpy()[~night]
    base_weight = cap / cap.mean()
    if context:
        a = np.column_stack([a, weather_context(x)[~night]])
        b = np.column_stack([b, weather_context(xe)[~ne]])
    if geometry:
        from round2_geometry import infer_geometry, projection_basis
        with threadpool_limits(limits=2, user_api="blas"):
            params = infer_geometry(past)
            a = np.column_stack([a, projection_basis(past.loc[~night], x.loc[~night], params)])
            b = np.column_stack([b, projection_basis(future.loc[~ne], xe.loc[~ne], params)])
    if recent:
        age = (past.dt.max()-past.dt.loc[~night]).dt.total_seconds().to_numpy()/86400
        base_weight *= 2**(-age/365.25)
        base_weight /= base_weight.mean()
    if nonlinear:
        with threadpool_limits(limits=2, user_api="blas"):
            a = np.column_stack([a, weather_basis(x.loc[~night])])
            b = np.column_stack([b, weather_basis(xe.loc[~ne])])
    if fleet:
        # Jointly estimate smooth changes in normalized yield. Limit the effective
        # extrapolation to one observed training range for short training windows.
        ref = np.mean(cap)
        def fleet_columns(table, capacities):
            dc = np.clip((capacities-ref)/ref, (cap.min()-ref)/ref,
                         (2*cap.max()-cap.min()-ref)/ref)
            ghi = table.all_ghi_mean.to_numpy()/1000
            clear = table.all_clearsky_mean.to_numpy()/1000
            kt = np.clip(ghi/(clear + 0.02), 0, 1.1)
            terms = [ghi, ghi*kt, clear*(1-kt), ghi*table.hh_sin, ghi*table.doy_cos]
            if regional_fleet:
                sites = columns(table, "ghi_dec")/1000
                terms.extend([sites-ghi[:,None], ghi*kt*table.doy_cos,
                              clear*(1-kt)*table.doy_cos, ghi*table.hh_cos])
            return np.column_stack(terms) * dc[:, None]
        a = np.column_stack([a, fleet_columns(x.loc[~night], cap)])
        b = np.column_stack([b, fleet_columns(xe.loc[~ne], cap_ev)])
    if difference_weight:
        # Learn the within-day ramp from observed consecutive half-hours as an
        # auxiliary task. Shared coefficients still predict levels without labels.
        times = past.dt.loc[~night]
        valid = (times.diff().dt.total_seconds().eq(1800) &
                 times.dt.normalize().eq(times.dt.normalize().shift())).to_numpy()[1:]
        a = np.vstack([a, np.diff(a, axis=0)[valid]])
        target = np.r_[target, np.diff(target)[valid]]
        base_weight = np.r_[base_weight, difference_weight*base_weight[1:][valid]]
    with threadpool_limits(limits=2, user_api="blas"):
        if fleet_penalty is not None:
            penalties = np.full(a.shape[1], penalty)
            penalties[-5:] = fleet_penalty
        else:
            penalties = penalty
        beta = robust_ridge(a, target, base_weight, penalties)
        pred = np.zeros(len(future))
        pred[~ne] = np.clip(b @ beta, 0, 1) * cap_ev
    return pred

"""Direct capacity-normalized regression using only the distributed weather fields."""
import numpy as np
import torch
from torch import nn
from features import RAD_CITIES

from round2_smooth import columns, fleet_power


def inputs(x):
    g = columns(x, "ghi_dec") / 1000
    clear = columns(x, "clearsky_dec") / 1000
    k = np.clip((g+0.01)/(clear+0.02), 0, 1.2)
    phase = 2*np.pi*x.doy.to_numpy()/365.25
    hour = np.pi*(x.hh.to_numpy()/2-12.25)/12
    seasonal = np.cos(phase - 2*np.pi*172/365.25)
    gh = columns(x, "ghi_h0") / 1000
    features = [g, gh, clear, k, np.sqrt(clear)*k**2, columns(x, "cloud_pct_dec")/100,
                (columns(x, "temp_c_dec")-12)/15, columns(x, "wind_ms_dec")/6,
                (columns(x, "rh_pct_dec")-75)/25]
    for lag in ("hm1", "hp1"):
        features += [columns(x, "ghi_"+lag)/1000-gh, columns(x, "kt_"+lag)-k]
    for lag in ("hm2", "hp2"):
        features += [(columns(x, "ghi_"+lag)/1000-gh).mean(1)]
    features += [seasonal, np.sin(phase), np.cos(hour), np.sin(hour), seasonal*np.cos(hour),
                 np.cos(2*hour), np.sin(2*hour), x.is_second_half.to_numpy(),
                 x.daily_kt.to_numpy(), x.daily_cloud_std.to_numpy()/25,
                 x.daily_temp_range.to_numpy()/10, x.temp_min_past24h.to_numpy()/15,
                 x.hrs_below_0c_past48h.to_numpy()/24]
    array = np.column_stack(features).astype(np.float32)
    return array, g.mean(1).astype(np.float32), np.sqrt(clear.mean(1)).astype(np.float32)


def train_seed(seed, a, b, y, weight, sky, sky_ev, base, base_ev):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(a.shape[1], 128), nn.SiLU(), nn.Dropout(0.05),
                          nn.Linear(128, 96), nn.SiLU(), nn.Dropout(0.05),
                          nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0014, weight_decay=0.012)
    # Fixed budget and tail averaging. No validation labels enter epoch selection.
    epochs = 72
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00008)
    snapshots = []
    for epoch in range(epochs):
        model.train()
        for rows in torch.randperm(len(a)).split(384):
            corr = model(a[rows]).flatten()
            pred = base[rows] + sky[rows] * corr
            losses = nn.functional.smooth_l1_loss(pred, y[rows], reduction="none", beta=0.009)
            loss = (losses * weight[rows]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        sched.step()
        if epoch in (55, 63, 71):
            model.eval()
            with torch.no_grad():
                snapshots.append((base_ev + sky_ev * model(b).flatten()).numpy())
    return np.mean(snapshots, axis=0)


def fit_net(name, x, night, past, xe, ne, future):
    powers = {"net_cf": 1.0, "net_fleet": 0.86, "net_solar": 0.86, "net_solar_adaptive": None}
    power = powers[name]
    if power is None:
        power = fleet_power(x, night, past)
    print(f"{name} training capacity elasticity={power:.6f}", flush=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a, base, sky = inputs(x.loc[~night])
    b, base_ev, sky_ev = inputs(xe.loc[~ne])
    if name.startswith("net_solar"):
        from round2_geometry import infer_geometry, projection_basis
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2, user_api="blas"):
            geometry = infer_geometry(past)
            def extra(frame, table, keep):
                physics = projection_basis(frame.loc[keep], table.loc[keep], geometry)
                # Initial blocks contain local diffuse, direct-normal, clear-sky
                # beam proxy, vertical cosine, and irradiance times cosine.
                n = len(RAD_CITIES)
                return np.column_stack([physics[:,:4*n], physics[:,5*n:6*n]]).astype(np.float32)
            a = np.column_stack([a, extra(past, x, ~night)])
            b = np.column_stack([b, extra(future, xe, ~ne)])
    mean, std = a.mean(axis=0, dtype=float), np.maximum(a.std(axis=0, dtype=float), 0.08)
    a = torch.from_numpy(np.clip((a-mean)/std, -8, 8).astype(np.float32))
    b = torch.from_numpy(np.clip((b-mean)/std, -8, 8).astype(np.float32))
    cap = past.installed_capacity_mwp.to_numpy()[~night]
    cap_ev = future.installed_capacity_mwp.to_numpy()[~ne]
    reference = np.median(cap)
    scale = reference*(cap/reference)**power
    scale_ev = reference*(cap_ev/reference)**power
    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32)
    args = (a, b, tensor(past.generation_mw.to_numpy()[~night]/scale), tensor(scale/scale.mean()),
            tensor(0.04 + sky), tensor(0.04 + sky_ev), tensor(base), tensor(base_ev))
    pred = np.zeros(len(future))
    estimate = np.mean([train_seed(seed, *args) for seed in (42, 137)], axis=0)
    pred[~ne] = np.clip(estimate * scale_ev, 0, cap_ev)
    return pred

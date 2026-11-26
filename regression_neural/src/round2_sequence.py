"""Whole-day weather profiles, with a shared temporal filter for neighboring half-hours."""
import numpy as np
import torch
from torch import nn

from round2_net import inputs
from round2_smooth import fleet_power


class DayProfile(nn.Module):
    def __init__(self, count):
        super().__init__()
        width = 48
        self.project = nn.Conv1d(count, width, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(width, width, 3, dilation=d, padding=d),
                          nn.SiLU(), nn.Dropout(0.06), nn.Conv1d(width, width, 1))
            for d in (1, 2, 4)
        ])
        self.head = nn.Sequential(nn.SiLU(), nn.Conv1d(width, 1, 1))

    def forward(self, x):
        x = self.project(x)
        for block in self.blocks:
            x = x + block(x)
        return self.head(x).squeeze(1)


def day_tensor(values):
    return torch.from_numpy(np.ascontiguousarray(values.reshape(-1, 48), dtype=np.float32))


def fit_sequence(name, x, night, past, xe, ne, future):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for frame in (past, future):
        assert frame.dt.is_monotonic_increasing and frame.dt.diff().dropna().dt.total_seconds().eq(1800).all()
        assert frame.groupby(frame.dt.dt.normalize()).size().eq(48).all()
    power = fleet_power(x, night, past) if name == "sequence_fleet" else 1.0
    print(f"{name} training capacity elasticity={power:.6f}", flush=True)
    cap = past.installed_capacity_mwp.to_numpy()
    cap_ev = future.installed_capacity_mwp.to_numpy()
    ref = np.median(cap)
    scale = ref*(cap/ref)**power
    scale_ev = ref*(cap_ev/ref)**power
    a, base, sky = inputs(x)
    b, base_ev, sky_ev = inputs(xe)
    mu, sd = a[~night].mean(0, dtype=float), np.maximum(a[~night].std(0, dtype=float), 0.08)
    def normalize(w):
        matrix = np.clip((w-mu)/sd, -8, 8).astype(np.float32)
        return torch.from_numpy(matrix.reshape(-1, 48, matrix.shape[1]).transpose(0, 2, 1).copy())
    a, b = normalize(a), normalize(b)
    base, base_ev = day_tensor(base), day_tensor(base_ev)
    sky, sky_ev = day_tensor(0.04+sky), day_tensor(0.04+sky_ev)
    y = day_tensor(past.generation_mw.to_numpy()/scale)
    w = day_tensor((~night) * scale/scale.mean())
    outputs = []
    for seed in (42, 137):
        torch.manual_seed(seed)
        model = DayProfile(a.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.015)
        epochs = 80
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00006)
        for epoch in range(epochs):
            model.train()
            for rows in torch.randperm(len(a)).split(24):
                pred = base[rows] + sky[rows] * model(a[rows])
                loss = nn.functional.smooth_l1_loss(pred, y[rows], beta=0.009, reduction="none")
                optimizer.zero_grad()
                (w[rows]*loss).mean().backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            sched.step()
            if epoch in (61, 70, 79):
                model.eval()
                with torch.no_grad():
                    outputs.append((base_ev + sky_ev*model(b)).numpy().reshape(-1))
    pred_ev = np.clip(np.mean(outputs, axis=0)*scale_ev, 0, cap_ev)
    pred_ev[ne] = 0
    return pred_ev

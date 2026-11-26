"""Fit candidates on expanding histories and save paired validation predictions."""
import argparse

import numpy as np
import pandas as pd

from features import build_fold_features
from round2_eval import FOLDS, ORIGINS, record, split
from round2_combine import BLENDS, historical_weight
from round2_smooth import fit_smooth
from utils import load_data


def predict(name, x, night, past, xe, ne, future):
    if name == "blend_previous":
        cutoff = future.dt.min()
        eligible = [f for f, (_, end) in FOLDS.items() if pd.Timestamp(end, tz="UTC") <= cutoff]
        members = BLENDS["blend_ramp"]
        weight = 0.5
        if eligible:
            fit, calib = split(past,eligible[-1])
            a, n, b, nb = build_fold_features(fit,calib)
            pair = [predict(m,a,n,fit,b,nb,calib) for m in members]
            weight = historical_weight(calib.generation_mw.to_numpy(),pair)
        pair = [predict(m,x,night,past,xe,ne,future) for m in members]
        return weight*pair[0] + (1-weight)*pair[1]
    if name in BLENDS:
        return np.mean([predict(m, x, night, past, xe, ne, future) for m in BLENDS[name]], axis=0)
    if name == "m2_end_fleet":
        from modeling import fit_predict_forward_lgb
        from round2_smooth import fleet_power
        base, _, _ = fit_predict_forward_lgb(x, night, past, xe, ne, future, list(x),
                                             mode="residual", add_phys_feat=True)
        ref = past.installed_capacity_mwp.to_numpy()[~night].mean()
        return base*(future.installed_capacity_mwp.to_numpy()/ref)**(fleet_power(x, night, past)-1)
    if name in ("ramp2", "ramp6", "ramp2_regional"):
        return fit_smooth(x, night, past, xe, ne, future, fleet=True, geometry=True, context=True,
                          penalty=1.0, difference_weight=6.0 if name == "ramp6" else 2.0,
                          regional_fleet=name == "ramp2_regional")
    if name in ("context_r4", "context_r16", "context_r4_fleet"):
        return fit_smooth(x, night, past, xe, ne, future, fleet=True, geometry=True, context=True,
                          penalty=16.0 if name == "context_r16" else 4.0,
                          fleet_penalty=0.15 if name == "context_r4_fleet" else 1.0)
    if name in ("smooth", "smooth_fleet", "kernel", "kernel_fleet", "kernel_fleet_recent",
                "solar", "solar_fleet", "solar_fleet_recent", "solar_fleet_reg",
                "solar_context_fleet_reg", "solar_context_fleet_reg_recent"):
        return fit_smooth(x, night, past, xe, ne, future, fleet="fleet" in name,
                          nonlinear=name.startswith("kernel"), recent=name.endswith("recent"),
                          geometry=name.startswith("solar"), context="context" in name,
                          penalty=1.0 if "_reg" in name else 0.09)
    if name in ("sequence_cf", "sequence_fleet"):
        from round2_sequence import fit_sequence
        return fit_sequence(name, x, night, past, xe, ne, future)
    from round2_net import fit_net
    return fit_net(name, x, night, past, xe, ne, future)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--folds", nargs="+", choices=list(ORIGINS), default=list(FOLDS))
    args = parser.parse_args()
    train, _, _ = load_data()
    for fold in args.folds:
        past, future = split(train, fold)
        x, night, xe, ne = build_fold_features(past, future)
        for name in args.models:
            pred = predict(name, x, night, past, xe, ne, future)
            record(name, fold, future, pred)


if __name__ == "__main__":
    main()

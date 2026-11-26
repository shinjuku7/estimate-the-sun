"""Paired local experiments using cached out-of-time component predictions."""
import numpy as np
import pandas as pd

from features import build_feature_table
from round2_eval import FOLDS, RESULTS, record, split
from round2_metrics import prediction
from round2_smooth import fleet_power
from utils import load_data


# Fixed equal averages. No weight is estimated on an evaluation period.
BLENDS = {
    "blend_net": ("solar_context_fleet_reg", "net_fleet"),
    "blend_sequence": ("solar_context_fleet_reg", "sequence_fleet"),
    "blend_m2": ("solar_context_fleet_reg", "m2_end_fleet"),
    "blend_ramp": ("ramp2", "net_fleet"),
    "blend_solar": ("ramp2", "net_solar"),
    "blend_adaptive": ("ramp2", "net_solar_adaptive"),
    "net_family": ("net_fleet", "net_solar", "net_solar_adaptive"),
    "blend_family": ("ramp2_regional", "net_family"),
}


def historical_weight(truth, predictions):
    # The MAE-minimizing convex weight is a weighted median. Shrink halfway
    # to equal weighting so a single season has limited influence.
    first, second = predictions
    delta = first-second
    keep = np.abs(delta)>1e-8
    ratio = (truth[keep]-second[keep])/delta[keep]
    weights = np.abs(delta[keep])
    if not len(ratio):
        return 0.5
    order = np.argsort(ratio)
    middle = np.searchsorted(np.cumsum(weights[order]), weights.sum()/2)
    return 0.25 + 0.5*np.clip(ratio[order[middle]],0,1)


def main():
    train, _, _ = load_data()
    weight = 0.5
    weights = []
    for fold in FOLDS:
        past, future = split(train, fold)
        x, night = build_feature_table(past)
        power = fleet_power(x, night, past)
        ref = past.installed_capacity_mwp.to_numpy()[~night].mean()
        def aligned(name):
            d = prediction(name, fold)
            assert d.id.is_unique and np.array_equal(d.id, future.id)
            return d.prediction.to_numpy()
        corrected = aligned("m2_end")*(future.installed_capacity_mwp.to_numpy()/ref)**(power-1)
        record("m2_end_fleet", fold, future, corrected)
        for name, members in BLENDS.items():
            record(name, fold, future, np.mean([aligned(m) for m in members], axis=0))
        pair = [aligned(m) for m in BLENDS["blend_ramp"]]
        record("blend_previous", fold, future, weight*pair[0]+(1-weight)*pair[1])
        weights.append((fold, weight))
        weight = historical_weight(future.generation_mw.to_numpy(), pair)
    weights.append(("full",weight))
    pd.DataFrame(weights,columns=["fold","ramp2_weight"]).to_csv(RESULTS/"prior_blend_weights.csv",index=False)


if __name__ == "__main__":
    main()

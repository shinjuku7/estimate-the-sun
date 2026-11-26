"""Summarize local rows only. All biases are prediction minus observation in MW."""
import argparse

import numpy as np
import pandas as pd

from features import build_feature_table
from round2_eval import FOLDS, RESULTS
from utils import load_data


def prediction(name, fold):
    return pd.read_csv(RESULTS / f"{name}_{fold}.csv.gz")


def diagnose(train):
    x, _ = build_feature_table(train)
    data = train[["id", "dt", "cf", "installed_capacity_mwp"]].copy()
    data["ghi_kw"] = x.all_ghi_mean/1000
    data["kt"] = x.all_ghi_mean / x.all_clearsky_mean.clip(lower=1)
    data["daylight"] = x.all_clearsky_mean > 100
    data["clear"] = data.daylight & (data.kt > 0.8)
    data["year"] = data.dt.dt.year
    data["month"] = data.dt.dt.month
    data["slot"] = data.dt.dt.hour*2 + data.dt.dt.minute//30
    key = ["year", "month"]
    rates = data.groupby(key).agg(daylight=("daylight","sum"), clear=("clear","sum"),
                                  capacity_mwp=("installed_capacity_mwp","mean"))
    sub = data.loc[data.clear]
    sums = sub.groupby(key)[["cf","ghi_kw"]].sum()
    rates["yield_ratio"] = sums.cf/sums.ghi_kw
    rates["clear_share"] = rates.clear/rates.daylight
    rates.to_csv(RESULTS/"clear_yield_month.csv")
    # Equal weight for identical UTC half-hour slots present in every year.
    noon = sub.loc[sub.month.isin([4,5,6]) & sub.slot.between(20,29)]
    slots = noon.groupby(["month","slot","year"]).agg(n=("id","size"),cf=("cf","sum"),g=("ghi_kw","sum"))
    slots["yield_ratio"] = slots.cf/slots.g
    valid = slots.yield_ratio.where(slots.n>=2).unstack("year").dropna()
    matched = valid.groupby(level="month").mean()
    matched["common_slots"] = valid.groupby(level="month").size()
    matched.reindex([4,5,6]).to_csv(RESULTS/"matched_clear_yield.csv")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("models", nargs="*")
    args = p.parse_args()
    if not args.models:
        args.models = sorted(path.name.removesuffix("_early.csv.gz")
                             for path in RESULTS.glob("*_early.csv.gz"))
    train, _, _ = load_data()
    weather = diagnose(train)
    blocks = []
    for name in args.models:
        for fold in FOLDS:
            pred = prediction(name, fold)
            row = pred.merge(weather[["id","dt","daylight","clear","kt"]], on="id", validate="one_to_one")
            assert len(row) == len(pred)
            err = row.prediction-row.generation_mw
            hour = row.dt.dt.hour
            row["model"], row["fold"] = name, fold
            row["error"], row["abs_error"] = err, err.abs()
            row["month"] = row.dt.dt.strftime("%Y-%m")
            row["date"] = row.dt.dt.normalize()
            row["utc_hours"] = pd.cut(hour, [-1,5,8,11,14,17,20,23],
                                      labels=["00-05","06-08","09-11","12-14","15-17","18-20","21-23"])
            row["sky"] = np.select([~row.daylight, row.clear], ["low_light","clear"], default="other_daylight")
            row["spring"] = np.where(row.dt.dt.month.isin([4,5,6]), "Apr-Jun", "other")
            row["capacity_gwp"] = (row.installed_capacity_mwp/1000).astype(int)
            blocks.append(row)
    all_rows = pd.concat(blocks, ignore_index=True)
    def metrics(by, name):
        result = all_rows.groupby(by, observed=True).agg(rows=("id","size"),
                         mae=("abs_error","mean"),bias=("error","mean")).reset_index()
        result.to_csv(RESULTS/f"{name}.csv", index=False, float_format="%.6f")
        return result
    scores = metrics(["model","fold"], "fold_metrics")
    metrics(["model","fold","month"], "monthly_metrics")
    metrics(["model","fold","utc_hours"], "clock_metrics")
    metrics(["model","fold","spring","sky"], "sky_metrics")
    metrics(["model","fold","capacity_gwp"], "capacity_metrics")
    pooled = metrics(["model"], "pooled_metrics")
    summary = scores.pivot(index="model",columns="fold",values="mae")
    summary["mean"] = summary.mean(axis=1)
    summary["worst"] = scores.groupby("model").mae.max()
    summary["pooled"] = pooled.set_index("model").mae
    summary = summary.sort_values("mean")
    summary.to_csv(RESULTS/"comparison.csv", float_format="%.6f")
    print(summary.round(3))
    daily = all_rows.groupby(["fold","date","model"]).abs_error.mean().unstack()
    intervals = []
    for fold, vals in daily.groupby(level=0):
        vals = vals.droplevel(0).sort_index()
        # Circular blocks preserve within-week correlations in this descriptive interval.
        rng = np.random.default_rng(42)
        starts = rng.integers(0,len(vals),size=(1500,int(np.ceil(len(vals)/7))))
        indices = ((starts[:,:,None]+np.arange(7)) % len(vals)).reshape(1500,-1)[:,:len(vals)]
        for name in args.models:
            delta = (vals[name]-vals["m2_end"]).to_numpy()
            low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            intervals.append((name,fold,delta.mean(),low,high))
    pd.DataFrame(intervals, columns=["model","fold","delta_mae","low","high"]).to_csv(
        RESULTS/"paired_week_intervals.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()

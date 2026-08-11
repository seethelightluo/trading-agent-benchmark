"""miner_2 exploration batch 2: skewness / drawdown / macro-conditional families.
Universe: 15 tradable cross-asset instruments. IC = cross-sectional Spearman rank IC.
Admission gates: |IC|>=0.007 and |ICIR|>=0.084 @ h=10.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_2_lib import (load_panel, load_macro, validate_factor,
                         WATCH, MAX_VISIBLE, FACTOR_LAST)

EPS = 1e-12
panel = load_panel()
rets = panel.pct_change()
mac = load_macro()


def skew_20():
    return rets.rolling(20).skew()


def drawdown_60():
    return panel / panel.rolling(60).max() - 1.0


def time_since_high_60():
    out = {}
    for s in WATCH:
        c = panel[s].dropna()
        rmax = c.rolling(60, min_periods=10).max()
        since = pd.Series(np.nan, index=c.index)
        # days since last time close hit its rolling max
        hit = (c >= rmax).astype(float)
        # count consecutive days since last hit
        groups = (hit == 0).cumsum()
        days = groups.groupby(groups).cumcount() + 1
        since = days.where(hit == 0, 0.0)
        out[s] = since
    return pd.DataFrame(out, index=panel.index)


def illiquidity_20():
    return (rets.abs() / (panel.rolling(20).mean().div(1) * 0 + panel)).mul(0)  # placeholder


def amihud_20():
    # |ret| / volume, averaged over 20d (per-asset)
    out = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        r = df["close"].pct_change()
        am = (r.abs() / (df["volume"] + EPS)).rolling(20).mean()
        out[s] = am
    return pd.DataFrame(out, index=panel.index)


def vol_ratio_10x60():
    return rets.rolling(10).std() / rets.rolling(60).std()


def usdjpy_beta_cond_60x20():
    usdjpy = mac["USDJPY"]
    usdjpyr = usdjpy.pct_change()
    beta = rets.rolling(60).cov(usdjpyr) / usdjpyr.rolling(60).var()
    cond = usdjpy / usdjpy.shift(20) - 1.0
    return beta * cond


def dxy_beta_cond_60x20():
    dxy = mac["DXY"]
    dxy_r = dxy.pct_change()
    beta = rets.rolling(60).cov(dxy_r) / dxy_r.rolling(60).var()
    cond = dxy / dxy.shift(20) - 1.0
    return beta * cond


candidates = {
    "skew_20": skew_20(),
    "drawdown_60": drawdown_60(),
    "time_since_high_60": time_since_high_60(),
    "amihud_20": amihud_20(),
    "vol_ratio_10x60": vol_ratio_10x60(),
    "usdjpy_beta_cond_60x20": usdjpy_beta_cond_60x20(),
    "dxy_beta_cond_60x20": dxy_beta_cond_60x20(),
}

for name, fdf in candidates.items():
    results = validate_factor(name, lambda p, m, fdf=fdf: fdf)
    print(f"  n_nan={int(fdf.isna().sum().sum())}")

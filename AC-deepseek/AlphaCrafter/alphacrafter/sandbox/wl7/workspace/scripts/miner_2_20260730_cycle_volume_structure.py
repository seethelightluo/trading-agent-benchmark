"""miner_2 exploration: volume & OHLC-structure factor family.
Universe: 15 tradable cross-asset instruments (daily, 2020-01-01..2026-07-15 warm-up window).
Uses per-asset trading calendars; IC = cross-sectional Spearman rank IC (>=8 assets).
Admission gates (benchmark contract): |IC|>=0.007 and |ICIR|>=0.084 @ h=10.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_2_lib import (load_panel, load_macro, validate_factor,
                         WATCH, MAX_VISIBLE, FACTOR_LAST)

EPS = 1e-12


def load_full_panel():
    cols = {s: {} for s in WATCH}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        cols[s]["open"] = df["open"].astype(float)
        cols[s]["high"] = df["high"].astype(float)
        cols[s]["low"] = df["low"].astype(float)
        cols[s]["close"] = df["close"].astype(float)
        cols[s]["volume"] = df["volume"].astype(float)
    idx = sorted(set().union(*[set(v["close"].index) for v in cols.values()]))
    idx = pd.DatetimeIndex(idx)
    out = {k: pd.DataFrame({s: {kk: cols[s][kk] for s in WATCH}} for kk in []) for k in []}
    opens = pd.DataFrame({s: cols[s]["open"].reindex(idx) for s in WATCH})
    highs = pd.DataFrame({s: cols[s]["high"].reindex(idx) for s in WATCH})
    lows = pd.DataFrame({s: cols[s]["low"].reindex(idx) for s in WATCH})
    closes = pd.DataFrame({s: cols[s]["close"].reindex(idx) for s in WATCH})
    vols = pd.DataFrame({s: cols[s]["volume"].reindex(idx) for s in WATCH})
    return opens, highs, lows, closes, vols


opens, highs, lows, closes, vols = load_full_panel()


def per_asset_df(fn):
    """Apply Series->Series fn on each asset's own calendar, reindex to union."""
    out = {}
    for s in WATCH:
        o = opens[s].dropna(); h = highs[s].dropna(); lo = lows[s].dropna()
        c = closes[s].dropna(); v = vols[s].dropna()
        out[s] = fn(o, h, lo, c, v)
    return pd.DataFrame(out, index=closes.index)


def vol_trend(short, long):
    def fn(o, h, lo, c, v):
        vs = v.rolling(short).mean()
        vl = v.rolling(long).mean()
        return np.log((vs + EPS) / (vl + EPS))
    return per_asset_df(fn)


def body_ratio(w):
    def fn(o, h, lo, c, v):
        rng = (h - lo).replace(0, np.nan)
        body = (c - o).abs() / (rng + EPS)
        return body.rolling(w).mean()
    return per_asset_df(fn)


def upper_wick(w):
    def fn(o, h, lo, c, v):
        rng = (h - lo).replace(0, np.nan)
        uw = (h - np.maximum(o, c)) / (rng + EPS)
        return uw.rolling(w).mean()
    return per_asset_df(fn)


def range_pos(w):
    def fn(o, h, lo, c, v):
        rng = h.rolling(w).max() - lo.rolling(w).min()
        return (c - lo.rolling(w).min()) / (rng + EPS)
    return per_asset_df(fn)


def volume_confirm_mom(mom_w, vol_short, vol_long):
    """10d momentum scaled by sign/level of volume expansion."""
    def fn(o, h, lo, c, v):
        mom = c / c.shift(mom_w) - 1.0
        vt = np.log((v.rolling(vol_short).mean() + EPS) / (v.rolling(vol_long).mean() + EPS))
        return mom * np.sign(vt)
    return per_asset_df(fn)


candidates = {
    "vol_trend_5x60": vol_trend(5, 60),
    "vol_trend_10x60": vol_trend(10, 60),
    "body_ratio_20": body_ratio(20),
    "upper_wick_20": upper_wick(20),
    "range_pos_20": range_pos(20),
    "vol_confirm_mom10": volume_confirm_mom(10, 5, 60),
}

for name, fdf in candidates.items():
    results = validate_factor(name, lambda panel, macro, fdf=fdf: fdf)
    print(f"  n_nan={int(fdf.isna().sum().sum())}  last_date={fdf.index.max().date()}")

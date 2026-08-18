"""Exploration sweep B (miner_3, 2026-08-12): autocorrelation, systematic-risk,
macro-sensitivity, risk-shape, and liquidity factor families.

These families are deliberately distinct from the existing library
(momentum skip-5, vol-of-vol, vix-beta-cond, rng_pos, skew, bbz) to add
diversity. Admission gate shared with the benchmark (15-instrument universe):
  abs(IC) >= 0.0070 and abs(ICIR) >= 0.0840 at h=10.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import (  # noqa: E402
    evaluate,
    load_closes,
    load_macro,
    weekday_grid,
)

closes = load_closes()
macro = load_macro()
print("assets:", len(closes), "macro:", list(macro.keys()))

# Equal-weight cross-asset market return on the common weekday grid (for beta)
grid = weekday_grid(closes)
ret_panel = pd.DataFrame({a: closes[a].reindex(grid).pct_change() for a in closes})
mkt = ret_panel.mean(axis=1, skipna=True)


def var_ratio(close, n=20, k=5):
    """Variance ratio of log prices: Var(n-day ret) / ((n/k) * Var(k-day ret)).
    >1 trending, <1 mean-reverting. Trend-persistence signal (distinct from
    price momentum itself)."""
    ln = np.log(close)
    rn = ln.diff(n)
    rk = ln.diff(k)
    return rn.rolling(n).var() / (rk.rolling(n).var() * (n / k))


def down_vol_ratio(close, n=60):
    """Downside deviation / total volatility: share of risk coming from losses."""
    r = close.pct_change()
    neg = r.where(r < 0)
    dvol = (neg ** 2).rolling(n, min_periods=max(10,n//3)).mean() ** 0.5
    vol = r.rolling(n).std().replace(0, np.nan)
    return dvol / vol


def gain_loss(close, n=60):
    """Up-day mean return / |down-day mean return| (quality/asymmetry)."""
    r = close.pct_change()
    pos = r.where(r > 0)
    neg = r.where(r < 0)
    up = pos.rolling(n, min_periods=max(10,n//3)).mean()
    dn = neg.rolling(n, min_periods=max(10,n//3)).mean().abs()
    return up / dn.replace(0, np.nan)


def vol_trend(close, ns=10, nl=60):
    """Short-horizon vol / long-horizon vol (volatility term-structure)."""
    r = close.pct_change()
    vs = r.rolling(ns).std()
    vl = r.rolling(nl).std().replace(0, np.nan)
    return vs / vl


def amihud(close, volume, n=60):
    """Amihud illiquidity: mean(|ret|/volume), log-scaled."""
    r = close.pct_change().abs()
    return np.log1p((r / volume.replace(0, np.nan)).rolling(n).mean())


def rolling_beta(x, y, n):
    cov = x.rolling(n).cov(y)
    var = y.rolling(n).var().replace(0, np.nan)
    return cov / var


def lowbeta_60(close):
    """Negative rolling beta vs equal-weight cross-asset market (low-beta tilt)."""
    x = close.reindex(grid).pct_change()
    return -rolling_beta(x, mkt, 60)


def dxy_beta_60(close):
    """Rolling beta of asset returns to DXY returns (USD-sensitivity)."""
    x = close.pct_change()
    d = macro["DXY"].reindex(x.index).pct_change()
    pair = pd.concat([x, d], axis=1).dropna()
    return rolling_beta(pair.iloc[:, 0], pair.iloc[:, 1], 60)


# volume tables (for Amihud)
vols = {}
for a in closes:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"]).set_index("date")
    vols[a] = df["volume"].astype(float)

candidates = {
    "var_ratio_20x5": {a: var_ratio(closes[a]) for a in closes},
    "lowbeta_60d": {a: lowbeta_60(closes[a]) for a in closes},
    "dxy_beta_60d": {a: dxy_beta_60(closes[a]) for a in closes},
    "down_vol_ratio_60d": {a: down_vol_ratio(closes[a]) for a in closes},
    "gain_loss_60d": {a: gain_loss(closes[a]) for a in closes},
    "vol_trend_10x60": {a: vol_trend(closes[a]) for a in closes},
    "amihud_60d": {a: amihud(closes[a], vols[a]) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()
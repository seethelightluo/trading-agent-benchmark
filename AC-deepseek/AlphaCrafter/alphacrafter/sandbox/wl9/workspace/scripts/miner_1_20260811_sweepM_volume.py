"""miner_1 (2026-08-11 revisit): sweep M - VOLUME dimension.

Current library is price-feature dominated (mom, bb_width, vol_z, rng_pos, skew,
kurt, beta_VIX, days_since_high, kaufman_eff, streak_len, dxy_corr). Volume is a
fresh orthogonal dimension almost entirely uncovered. Probe volume-trend,
volume-price alignment, OBV momentum, and volume-surge normalization.

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; prefer max lib corr<0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro
from miner3_20260730_harness import DATA_DIR, STOCK_DIR


def load_closes_vol():
    closes, vols = {}, {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        if "volume" not in df.columns:
            continue
        s = df.set_index("date")
        closes[a] = s["close"].astype(float)
        vols[a] = s["volume"].astype(float)
    return closes, vols


def vol_trend(v, short=5, long=60):
    """Volume ratio short/long mean: recent activity vs baseline (liquidity surge)."""
    sv = v.rolling(short, min_periods=3).mean()
    lv = v.rolling(long, min_periods=30).mean().replace(0, np.nan)
    return sv / lv


def obv_mom(close, vol, n=20):
    """On-Balance-Volume momentum normalized by price - trend confirmation."""
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    return obv.pct_change(n).replace([np.inf, -np.inf], np.nan)


def vol_price_corr(close, vol, n=20):
    """Rolling corr between price change sign and volume (buying pressure proxy)."""
    r = close.pct_change()
    d = vol.pct_change()
    return r.rolling(n, min_periods=10).corr(d)


def vol_z_score(v, n=60):
    """Standardized volume deviation from its trailing baseline (abnormal volume)."""
    mean = v.rolling(n, min_periods=30).mean()
    std = v.rolling(n, min_periods=30).std(ddof=0).replace(0, np.nan)
    return (v - mean) / std


def volume_surge_price(close, vol, n=20):
    """Price change over n days / volume surge - price gained on above-normal volume."""
    vs = (vol / vol.rolling(n, min_periods=10).mean()).replace(0, np.nan)
    return close.pct_change(n) * vs


closes, vols = load_closes_vol()
macro = load_macro()
print("closes:", len(closes), "vols:", len(vols))

candidates = {
    "vol_trend_5x60": {a: vol_trend(vols[a], 5, 60) for a in vols},
    "obv_mom_20": {a: obv_mom(closes[a], vols[a], 20) for a in vols},
    "vol_price_corr_20": {a: vol_price_corr(closes[a], vols[a], 20) for a in vols},
    "vol_z_60": {a: vol_z_score(vols[a], 60) for a in vols},
    "vol_surge_20": {a: volume_surge_price(closes[a], vols[a], 20) for a in vols},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()
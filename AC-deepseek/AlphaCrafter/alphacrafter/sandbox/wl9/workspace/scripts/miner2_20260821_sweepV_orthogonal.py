"""miner_2 (2026-08-21): explore volume/breadth/yield fresh orthogonal dimensions.

Library already covers momentum, vol, skew/kurt, range position, autocorr, beta.
Target structurally fresh dims with low correlation to the library (<0.5):
  - on-balance-volume (OBV) momentum
  - price-volume correlation
  - volume expansion ratio (volume z)
  - cross-asset breadth (fraction of assets in uptrend)
  - yield-curve slope (US10Y - CN10Y regime)
  - realized vol percentile vs cross-section
Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; max_abs_library_correlation < 0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, STOCK_DIR, evaluate, load_closes

closes = load_closes()


def load_volume():
    out = {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")["volume"].astype(float)
    return out


def obv_mom(close, vol, n=10):
    """OBV slope over n days (volume-weighted trend confirmation), normalized."""
    obv = (np.sign(close.diff().fillna(0)) * vol).cumsum()
    return obv.diff(n) / vol.rolling(n, min_periods=5).sum().replace(0, np.nan)


def price_vol_corr(close, vol, n=20):
    """Rolling correlation of daily return with volume change (price-volume linkage)."""
    r = close.pct_change()
    dv = vol.pct_change()
    return r.rolling(n, min_periods=10).corr(dv)


def vol_expand(close, vol, n=10, base=120):
    """Short volume / long volume - 1 (volume expansion regime)."""
    vs = vol.rolling(n, min_periods=5).mean()
    vl = vol.rolling(base, min_periods=60).mean()
    return vs / vl.replace(0, np.nan) - 1.0


def cross_breadth(close, n=10):
    """Fraction of the 15-name universe in a 10d uptrend (cross-sectional breadth)."""
    up = pd.DataFrame({a: (close[a] > close[a].shift(n)) for a in close})
    return up.mean(axis=1)


def yield_slope(close):
    """US10Y - CN10Y spread as yield-curve regime signal."""
    a = close["US10Y"] - close["CN10Y"]
    return a


def rv_zscore(close, n=20, base=120):
    """Realized vol vs its own trailing baseline (vol regime z)."""
    rv = close.pct_change().rolling(n, min_periods=10).std()
    mu = rv.rolling(base, min_periods=60).mean()
    sd = rv.rolling(base, min_periods=60).std()
    return (rv - mu) / sd.replace(0, np.nan)


vol = load_volume()
cands = {}
cands["obv_mom_10"] = {a: obv_mom(closes[a], vol[a], 10) for a in closes if a in vol}
cands["price_vol_corr_20"] = {a: price_vol_corr(closes[a], vol[a], 20) for a in closes if a in vol}
cands["vol_expand_10_120"] = {a: vol_expand(closes[a], vol[a], 10, 120) for a in closes if a in vol}
cands["cross_breadth_10"] = cross_breadth(closes, 10)
cands["yield_slope"] = yield_slope(closes)
cands["rv_zscore_20_120"] = {a: rv_zscore(closes[a], 20, 120) for a in closes}

for name, vals in cands.items():
    try:
        res = evaluate(closes, vals, name, horizon=10)
        print(f"RESULT {name}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
              f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}\n")
    except Exception as e:
        print(f"ERROR {name}: {e}\n")

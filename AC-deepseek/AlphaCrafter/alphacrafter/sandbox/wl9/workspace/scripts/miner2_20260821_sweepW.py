"""
miner_2 (2026-08-21): broad sweep of orthogonal candidates that pass gate with
library corr < 0.5. Try to find a vol-momentum interplay using denominators
less correlated with rng_pos/momentum, plus reversal and volume-permutation
ideas that are structurally distinct.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import evaluate, load_closes  # noqa: E402

ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def load_full_ohlc():
    out = {}
    for a in ASSETS:
        f = f"../persistent/stock_data/{a}.csv"
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out


def vol_scaled_short_mom(close, short=10, voln=30, skip=3):
    mom = close.shift(skip) / close.shift(skip + short) - 1.0
    sd = close.pct_change().rolling(voln, min_periods=voln // 2).std()
    return mom / sd.replace(0, np.nan)


def vol_scaled_reversal(close, short=5, voln=20):
    rev = close.shift(1) / close.shift(1 + short) - 1.0
    sd = close.pct_change().rolling(voln, min_periods=voln // 2).std()
    return -rev / sd.replace(0, np.nan)  # negative -> mean reversion


def zscore_ret_pos(close, n=10):
    """Z-score of recent return vs its own history (position within own distribution)."""
    r = close.pct_change()
    mom = close / close.shift(n) - 1.0
    hist = (close / close.shift(n)).rolling(250, min_periods=100)
    mu = hist.mean()
    sd = hist.std()
    return (mom - mu) / sd.replace(0, np.nan)


def vol_adj_mom_ratio(close, short=10, med=40, voln=30, skip=5):
    """Momentum ratio scaled by vol (attempt to orthogonalize vs rng_pos)."""
    ms = close.shift(skip) / close.shift(skip + short) - 1.0
    mm = close.shift(skip) / close.shift(skip + med) - 1.0
    sd = close.pct_change().rolling(voln, min_periods=voln // 2).std()
    return (ms / mm.replace(0, np.nan)) / sd.replace(0, np.nan)


def volume_price_trend(close, vol, n=20):
    """VPT: cumulative sum of ret*volume, then z-scored. Volume-price interplay."""
    r = close.pct_change().fillna(0.0)
    vpt = (r * vol).rolling(n, min_periods=n // 2).sum()
    return vpt


def wtd_mom_accel(close, short=10, med=40, skip=5):
    """Difference of two speed momentum (acceleration) - distinct from ratio."""
    ms = close.shift(skip) / close.shift(skip + short) - 1.0
    mm = close.shift(skip) / close.shift(skip + med) - 1.0
    return ms - mm


def main():
    closes = load_closes()
    ohlc = load_full_ohlc()
    vol = {a: ohlc[a]["volume"].astype(float) for a in closes if a in ohlc}

    cands = {}
    cands["vol_scaled_short_mom_10_30"] = {a: vol_scaled_short_mom(s, 10, 30, 3) for a, s in closes.items()}
    cands["vol_scaled_reversal_5_20"] = {a: vol_scaled_reversal(s, 5, 20) for a, s in closes.items()}
    cands["zscore_ret_pos_10"] = {a: zscore_ret_pos(s, 10) for a, s in closes.items()}
    cands["vol_adj_mom_ratio_10_40"] = {a: vol_adj_mom_ratio(s, 10, 40, 30, 5) for a, s in closes.items()}
    cands["volume_price_trend_20"] = {a: volume_price_trend(closes[a], vol[a], 20) for a in closes if a in vol}
    cands["wtd_mom_accel_10_40"] = {a: wtd_mom_accel(s, 10, 40, 5) for a, s in closes.items()}

    for name, vals in cands.items():
        try:
            res = evaluate(closes, vals, name, horizon=10)
            print(f"RESULT {name}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
                  f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}")
        except Exception as e:
            print(f"ERROR {name}: {e}")


if __name__ == "__main__":
    main()

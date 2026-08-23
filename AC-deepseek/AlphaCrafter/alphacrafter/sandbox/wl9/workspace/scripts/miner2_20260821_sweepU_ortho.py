"""
miner_2 (2026-08-21): explore fresh orthogonal factor ideas.

Goal: find candidates passing IC/ICIR gate with low max_abs_library_correlation
(< 0.5) to avoid eviction. Existing library already covers momentum, vol, skew/
kurt, autocorr, beta, days_since_high. Focus on volume, range-efficiency, gap,
and cross-window dispersion that are structurally distinct.
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


def vol_efficiency(close, n=60):
    path = close.pct_change().abs().rolling(n, min_periods=n // 2).sum()
    net = (close / close.shift(n) - 1.0).abs()
    return net / path.replace(0, np.nan)


def amihud(close, vol, n=20):
    r = close.pct_change().abs()
    return (r / vol.replace(0, np.nan)).rolling(n, min_periods=n // 2).mean()


def vol_trend_ratio(close, vol, short=20, long=120):
    vs = vol.rolling(short, min_periods=short // 2).mean()
    vl = vol.rolling(long, min_periods=long // 2).mean()
    return vs / vl.replace(0, np.nan)


def hi_lo_eff(close, hi, lo, n=60):
    lo_mean = lo.rolling(n, min_periods=n // 2).mean()
    hi_mean = hi.rolling(n, min_periods=n // 2).mean()
    return (close - lo_mean) / (hi_mean - lo_mean).replace(0, np.nan)


def ret_vol_ratio(close, n=20):
    r = close.pct_change()
    mom = close / close.shift(n + 5) - 1.0  # skip 5
    sd = r.rolling(n, min_periods=n // 2).std() * np.sqrt(n)
    return mom / sd.replace(0, np.nan)


def main():
    closes = load_closes()
    ohlc = load_full_ohlc()
    vol = {a: ohlc[a]["volume"].astype(float) for a in closes if a in ohlc}
    hi = {a: ohlc[a]["high"].astype(float) for a in closes if a in ohlc}
    lo = {a: ohlc[a]["low"].astype(float) for a in closes if a in ohlc}

    cands = {}
    cands["vol_efficiency_60"] = {a: vol_efficiency(s, 60) for a, s in closes.items()}
    cands["amihud_20"] = {a: amihud(closes[a], vol[a], 20) for a in closes if a in vol}
    cands["vol_trend_ratio_20_120"] = {a: vol_trend_ratio(closes[a], vol[a], 20, 120) for a in closes if a in vol}
    cands["hi_lo_eff_60"] = {a: hi_lo_eff(closes[a], hi[a], lo[a], 60) for a in closes if a in hi}
    cands["ret_vol_ratio_20_skip5"] = {a: ret_vol_ratio(s, 20) for a, s in closes.items()}

    for name, vals in cands.items():
        try:
            res = evaluate(closes, vals, name, horizon=10)
            print(f"RESULT {name}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
                  f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}")
        except Exception as e:
            print(f"ERROR {name}: {e}")


if __name__ == "__main__":
    main()
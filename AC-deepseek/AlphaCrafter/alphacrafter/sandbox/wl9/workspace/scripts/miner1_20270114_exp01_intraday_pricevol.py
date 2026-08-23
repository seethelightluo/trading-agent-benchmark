"""miner_1 (2027-01-14): explore intraday price-volume structure candidates.

Current date 2027-01-14, visible through last completed trading day.
Validate on warm-up + recent window (all data through VISIBLE_END). Aim: find
fresh factor ideas with low library correlation and a plausible narrative.

Candidates:
  - carry_skew_60: 60d risk-adjusted upward bias (mean/median daily return) -> directional quality
  - hi_lo_cluster_20: fraction of days closing in the upper third of the day's range (up-capture)
  - wick_ratio_20: upper wick / lower wick balance (rejection at highs vs support at lows)
  - morning_gap_20: avg overnight-gap direction scaled by volatility
  - retracement_efficiency: how often closes retrace from intraday extremes
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import evaluate  # noqa: E402

ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
VISIBLE_END = "2027-01-13"   # previous completed trading day
STOCK_DIR = "../persistent/stock_data"


def load_ohlc():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"{STOCK_DIR}/{a}.csv", parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END].set_index("date")
        out[a] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return out


def carry_skew_60(s, ohlc, n=60):
    dret = s.pct_change()
    # directional quality: use sign-dampened magnitude -> upward bias per unit vol
    mean = dret.rolling(n, min_periods=30).mean()
    std = dret.rolling(n, min_periods=30).std()
    return (mean / std.replace(0, np.nan)).rolling(5).mean()


def hi_lo_cluster_20(ohlc, n=20):
    hi = ohlc["high"]; lo = ohlc["low"]; cl = ohlc["close"]
    rng = (hi - lo).replace(0, np.nan)
    pos = (cl - lo) / rng
    # rolling fraction of days in upper third of range
    upper = (pos > 0.667).astype(float).rolling(n, min_periods=10).mean()
    return upper - 0.5


def wick_ratio_20(ohlc, n=20):
    hi = ohlc["high"]; lo = ohlc["low"]; op = ohlc["open"]; cl = ohlc["close"]
    body = (cl - op).abs()
    upper_wick = hi - np.maximum(op, cl)
    lower_wick = np.minimum(op, cl) - lo
    tot = (body + upper_wick + lower_wick).replace(0, np.nan)
    balance = (upper_wick - lower_wick) / tot
    return balance.rolling(n, min_periods=10).mean()


def didnn_decode(s, vix):
    """Which side carried the session: close vs midrange, adjusted for prior-day close."""
    return s.pct_change().rolling(20).std() / vix.pct_change().rolling(20).std().replace(0, np.nan)


def trend_internal_20(s, ohlc, n=20):
    """Signed distance of close from rolling midrange, smoothed (internal strength)."""
    hi = ohlc["high"]; lo = ohlc["low"]
    mid = (hi + lo) / 2.0
    signed = (s - mid) / mid.replace(0, np.nan)
    return signed.rolling(n, min_periods=10).mean()


def main():
    ohlc = load_ohlc()
    closes = {a: ohlc[a]["close"] for a in ohlc}

    cands = {}
    cands["carry_skew_60"] = {a: carry_skew_60(s, ohlc[a]) for a, s in closes.items()}
    cands["hi_lo_cluster_20"] = {a: hi_lo_cluster_20(ohlc[a]) for a in ohlc}
    cands["wick_ratio_20"] = {a: wick_ratio_20(ohlc[a]) for a in ohlc}
    cands["trend_internal_20"] = {a: trend_internal_20(s, ohlc[a]) for a, s in closes.items()}

    for name, vals in cands.items():
        try:
            res = evaluate(closes, vals, name, horizon=10)
            print(f"RESULT {name}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
                  f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"ERROR {name}: {e}")


if __name__ == "__main__":
    main()
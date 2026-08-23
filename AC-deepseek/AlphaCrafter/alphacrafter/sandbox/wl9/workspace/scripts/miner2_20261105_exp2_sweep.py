"""miner_2 (2026-11-05): second sweep of orthogonal factor candidates.
Explore: overnight_ratio, tail_ratio, vol_term_slope, mom_accel, rsi_short.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner3_20260730_harness import evaluate  # noqa: E402

ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
VISIBLE_END = "2026-11-04"
DATA_DIR = Path("../persistent/stock_data")


def load_closes():
    closes = {}
    for a in ASSETS:
        f = DATA_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END]
        closes[a] = df.set_index("date")["close"].astype(float)
    return closes


def load_ohlc():
    out = {}
    for a in ASSETS:
        f = DATA_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= VISIBLE_END].set_index("date")
        out[a] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return out


def overnight_ratio(s, ohlc, n=20):
    o = ohlc["open"]
    prev_close = s.shift(1)
    gap = (o - prev_close) / prev_close.replace(0, np.nan)
    day = (s - o) / o.replace(0, np.nan)
    return gap.rolling(n, min_periods=10).mean()


def tail_ratio(s, ohlc, n=20):
    hi = ohlc["high"]; lo = ohlc["low"]
    up = hi.rolling(n, min_periods=10).max() - s
    dn = s - lo.rolling(n, min_periods=10).min()
    return (up - dn) / (up + dn).replace(0, np.nan)


def vol_term_slope(s, ohlc):
    hi = ohlc["high"]; lo = ohlc["low"]
    rng = (hi - lo) / s.replace(0, np.nan)
    v20 = rng.rolling(20, min_periods=10).mean()
    v10 = rng.rolling(10, min_periods=5).mean()
    v60 = rng.rolling(60, min_periods=30).mean()
    return (v60 - v20) / v60.replace(0, np.nan)


def mom_accel(s, n=40):
    mom_short = s.shift(5) / s.shift(15) - 1.0
    mom_long = s.shift(5) / s.shift(n + 5) - 1.0
    return mom_short - mom_long


def rsi_short(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0).rolling(n, min_periods=7).mean()
    loss = (-d.clip(upper=0)).rolling(n, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def main():
    closes = load_closes()
    ohlc = load_ohlc()
    cands = {}
    cands["overnight_ratio_20"] = {a: overnight_ratio(s, ohlc[a]) for a, s in closes.items() if a in ohlc}
    cands["tail_ratio_20"] = {a: tail_ratio(s, ohlc[a]) for a, s in closes.items() if a in ohlc}
    cands["vol_term_slope_shortflat"] = {a: vol_term_slope(s, ohlc[a]) for a, s in closes.items() if a in ohlc}
    cands["mom_accel_40"] = {a: mom_accel(s) for a, s in closes.items()}
    cands["rsi_short_14"] = {a: rsi_short(s) for a, s in closes.items()}

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

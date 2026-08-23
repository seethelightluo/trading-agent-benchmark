"""miner_2 (2026-11-05): explore fresh orthogonal factor candidates on recent data.

Current date 2026-11-05, visible through 2026-11-04. Validate on warm-up +
recent window. Goal: find factor ideas with low library correlation and a
plausible economic narrative. Candidates:
  - sharpe_60: realized risk-adjusted momentum (mean/std of ret over 60d)
  - drawdown_120: depth of current retracement from trailing high
  - volume_surge_20: recent volume z-score vs trailing norm
  - range_drift_ratio: persistence of intra-day range (vol regime regime)
  - cross_alpha_60: asset momentum relative to market median
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


def sharpe_60(s, n=60):
    r = s.pct_change()
    mean = r.rolling(n, min_periods=30).mean()
    std = r.rolling(n, min_periods=30).std()
    return (mean / std.replace(0, np.nan)).rolling(5).mean()


def drawdown_120(s, n=120):
    peak = s.rolling(n, min_periods=60).max()
    return s / peak - 1.0


def vol_surge_20(s, ohlc, n=20, norm=60):
    v = ohlc["volume"]
    v_norm = v.rolling(norm, min_periods=30).mean()
    v_std = v.rolling(norm, min_periods=30).std()
    z = (v.rolling(n, min_periods=10).mean() - v_norm) / v_std.replace(0, np.nan)
    return z


def range_drift_ratio(s, ohlc):
    hi = ohlc["high"]; lo = ohlc["low"]
    rng = (hi - lo) / s.replace(0, np.nan)
    r20 = rng.rolling(20, min_periods=10).mean()
    r60 = rng.rolling(60, min_periods=30).mean()
    return (r20 - r60) / r60.replace(0, np.nan)


def cross_alpha(closes, s, a, n=60):
    mom = s.shift(5) / s.shift(n + 5) - 1.0
    med = {}
    for b, sb in closes.items():
        mb = sb.shift(5) / sb.shift(n + 5) - 1.0
        med[b] = mb
    med_df = pd.DataFrame(med)
    cross_med = med_df.median(axis=1, skipna=True)
    return mom - cross_med.reindex(mom.index)


def main():
    closes = load_closes()
    ohlc = load_ohlc()
    cands = {}
    cands["sharpe_60"] = {a: sharpe_60(s) for a, s in closes.items()}
    cands["drawdown_120"] = {a: drawdown_120(s) for a, s in closes.items()}
    cands["volume_surge_20"] = {a: vol_surge_20(s, ohlc[a]) for a, s in closes.items() if a in ohlc}
    cands["range_drift_ratio_20v60"] = {a: range_drift_ratio(s, ohlc[a]) for a, s in closes.items() if a in ohlc}
    cands["cross_alpha_60"] = {a: cross_alpha(closes, s, a) for a, s in closes.items()}

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

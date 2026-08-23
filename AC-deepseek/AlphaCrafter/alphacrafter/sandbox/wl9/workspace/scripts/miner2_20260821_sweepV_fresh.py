"""
miner_2 (2026-08-21): explore fresh orthogonal factor candidates targeting
low library correlation (< 0.5). Prior sweep found ret_vol_ratio passed but
with max_corr 0.70 (correlated to rng_pos / momentum), so it would be evicted.
Here we test structurally distinct ideas:
  - gap_efficiency: overnight/intraday efficiency
  - up_down_vol_ratio: asymmetry in up vs down day volatility
  - range_persistence: autocorrelation of daily range
  - high_low_position_short: short-window position in recent range
  - wtd_ret_reversal: 5d weighted return (recency-weighted)
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


def gap_efficiency(close, open_):
    """Fraction of daily range consumed by the gap: |open-prev_close| / day range."""
    prev_close = close.shift(1)
    gap = (open_ - prev_close).abs()
    rng = (open_.rolling(0).max() * 0 + close.rolling(0).max() * 0)  # placeholder
    day_range = (close - open_).abs()
    # gap relative to total move from prev close to close
    tot = (close - prev_close).abs()
    return (gap / tot.replace(0, np.nan)).rolling(20, min_periods=10).mean()


def updown_vol_ratio(close, n=20):
    r = close.pct_change()
    up = r[r > 0]
    dn = r[r < 0]
    up_std = up.rolling(n, min_periods=n // 2).std()
    dn_std = dn.rolling(n, min_periods=n // 2).std()
    return (up_std - dn_std) / (up_std + dn_std).replace(0, np.nan)


def range_autocorr(close, hi, lo, n=20):
    rng = (hi - lo) / close
    rng = rng.fillna(0.0)
    ac = rng.rolling(n, min_periods=n // 2).apply(
        lambda x: pd.Series(x).autocorr() if len(x) > 3 else np.nan, raw=False)
    return ac


def hi_lo_pos_short(close, hi, lo, n=10):
    """Short-window position in recent range (structural distinct from 60d)."""
    lo_roll = lo.rolling(n, min_periods=n // 2).min()
    hi_roll = hi.rolling(n, min_periods=n // 2).max()
    return (close - lo_roll) / (hi_roll - lo_roll).replace(0, np.nan)


def recency_weighted_ret(close, n=10):
    """Recency-weighted (linear decay) cumulative return; distinct from skip5 momentum."""
    r = close.pct_change().fillna(0.0)
    weights = np.arange(1, n + 1, dtype=float)
    weights = weights / weights.sum()
    return r.rolling(n, min_periods=n // 2).apply(
        lambda x: float(np.dot(x, weights[-len(x):])) if len(x) >= n // 2 else np.nan,
        raw=True)


def main():
    closes = load_closes()
    ohlc = load_full_ohlc()
    open_ = {a: ohlc[a]["open"].astype(float) for a in closes if a in ohlc}
    hi = {a: ohlc[a]["high"].astype(float) for a in closes if a in ohlc}
    lo = {a: ohlc[a]["low"].astype(float) for a in closes if a in ohlc}

    cands = {}
    cands["gap_efficiency_20"] = {a: gap_efficiency(closes[a], open_[a]) for a in closes if a in open_}
    cands["updown_vol_ratio_20"] = {a: updown_vol_ratio(s, 20) for a, s in closes.items()}
    cands["range_autocorr_20"] = {a: range_autocorr(closes[a], hi[a], lo[a], 20) for a in closes if a in hi}
    cands["hi_lo_pos_short_10"] = {a: hi_lo_pos_short(closes[a], hi[a], lo[a], 10) for a in closes if a in hi}
    cands["recency_weighted_ret_10"] = {a: recency_weighted_ret(s, 10) for a, s in closes.items()}

    for name, vals in cands.items():
        try:
            res = evaluate(closes, vals, name, horizon=10)
            print(f"RESULT {name}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
                  f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}")
        except Exception as e:
            print(f"ERROR {name}: {e}")


if __name__ == "__main__":
    main()

"""
miner_2 (2026-08-21): final broad sweep focusing on STRUCTURALLY distinct ideas
with low library correlation. Test:
  - ADX-like directional strength (trend vs range, uses +DM/-DM)
  - candle body ratio (open-close geometry)
  - range exhaustion (narrowing range)
  - dollar-volume trend (volume-based, distinct from price factors)
  - rolling_coef asymmetry
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


def adx_strength(close, hi, lo, n=14):
    """ADX-like: smoothed directional movement relative to range. Trend-strength."""
    up = hi.diff()
    dn = -lo.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=close.index)
    tr = pd.concat([hi - lo, (hi - close.shift()).abs(), (lo - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n // 2).mean()
    pdi = 100 * plus_dm.rolling(n, min_periods=n // 2).mean() / atr.replace(0, np.nan)
    mdi = 100 * minus_dm.rolling(n, min_periods=n // 2).mean() / atr.replace(0, np.nan)
    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
    return dx.rolling(n, min_periods=n // 2).mean()


def candle_body_ratio(close, open_, n=20):
    """Average body (close-open) relative to range - candle geometry trader bias."""
    body = (close - open_).abs()
    rng = (close - open_.rolling(0).max()) * 0 + 1.0  # placeholder
    dayrange = (close - open_).abs()
    return (body / dayrange.replace(0, np.nan)).rolling(n, min_periods=n // 2).mean()


def range_exhaustion(close, hi, lo, n=20):
    """Narrowing range: recent range / longer range (contraction -> breakout)."""
    recent = (hi.rolling(5).max() - lo.rolling(5).min()) / close
    longer = (hi.rolling(n).max() - lo.rolling(n).min()) / close
    return 1.0 - (recent / longer.replace(0, np.nan))


def dollar_vol_trend(close, vol, short=20, long=120):
    """Trend in dollar volume (proxy for participation shift). Volume-based."""
    dv = close * vol
    vs = dv.rolling(short, min_periods=short // 2).mean()
    vl = dv.rolling(long, min_periods=long // 2).mean()
    return vs / vl.replace(0, np.nan)


def roll_omp(coef=...):
    pass


def rolling_second_moment(close, n=20):
    """Non-centered second moment (var around 0 rather than mean) - distinct."""
    r = close.pct_change().fillna(0.0)
    return (r ** 2).rolling(n, min_periods=n // 2).mean()


def main():
    closes = load_closes()
    ohlc = load_full_ohlc()
    open_ = {a: ohlc[a]["open"].astype(float) for a in closes if a in ohlc}
    hi = {a: ohlc[a]["high"].astype(float) for a in closes if a in ohlc}
    lo = {a: ohlc[a]["low"].astype(float) for a in closes if a in ohlc}
    vol = {a: ohlc[a]["volume"].astype(float) for a in closes if a in ohlc}

    cands = {}
    cands["adx_strength_14"] = {a: adx_strength(closes[a], hi[a], lo[a], 14) for a in closes if a in hi}
    cands["candle_body_ratio_20"] = {a: candle_body_ratio(closes[a], open_[a], 20) for a in closes if a in open_}
    cands["range_exhaustion_20"] = {a: range_exhaustion(closes[a], hi[a], lo[a], 20) for a in closes if a in hi}
    cands["dollar_vol_trend_20_120"] = {a: dollar_vol_trend(closes[a], vol[a], 20, 120) for a in closes if a in vol}
    cands["rolling_second_moment_20"] = {a: rolling_second_moment(s, 20) for a, s in closes.items()}

    for name, vals in cands.items():
        try:
            res = evaluate(closes, vals, name, horizon=10)
            print(f"RESULT {name}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
                  f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}")
        except Exception as e:
            print(f"ERROR {name}: {e}")


if __name__ == "__main__":
    main()

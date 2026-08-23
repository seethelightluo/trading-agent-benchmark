"""
miner_2 (2026-08-21): push for orthogonal candidates with lower library corr.
Prior vol-scaled mom passed but corr 0.61 (correlated to rng_pos/momentum).
Try structurally-distinct ideas:
  - downside/upside semi-deviation ratio
  - long-vol vs short-vol momentum differential
  - 3-mo seasonal momentum (calendar month return)
  - idiosyncratic residual vol after market-factor regression (needs macro)
  - half-life / range half-life
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


def semi_dev_ratio(close, n=20):
    """(downside semi-dev) / (upside semi-dev) - asymmetry of risk."""
    r = close.pct_change()
    down = r[r < 0]
    up = r[r > 0]
    dd = down.rolling(n, min_periods=n // 2).std()
    ud = up.rolling(n, min_periods=n // 2).std()
    return dd / ud.replace(0, np.nan)


def long_short_vol_diff(close, short=10, long=60):
    """(long vol - short vol) normalized - vol term-structure shift, uses diff not ratio."""
    sv = close.pct_change().rolling(short, min_periods=short // 2).std()
    lv = close.pct_change().rolling(long, min_periods=long // 2).std()
    return (lv - sv) / (lv + sv).replace(0, np.nan)


def monthly_seasonal(close, n=1):
    """Return over trailing ~21 trading days' ending month segment. Simple probe."""
    return close / close.shift(21) - 1.0


def rolling_skew3(close, n=20):
    """Skew using 3rd moment scaled - distinct from library skew_20d? use different win."""
    r = close.pct_change()
    return r.rolling(n, min_periods=n // 2).skew()


def kayc_eff(close, n=20):
    """Kaufman efficiency at different window (already have kaufman_eff_20d) - test 40d variant."""
    path = close.diff().abs().rolling(n, min_periods=n // 2).sum()
    net = (close - close.shift(n)).abs()
    return net / path.replace(0, np.nan)


def downs_dev_ratio(close, n=60):
    """Return / downside deviation (Sortino-like); uses 60d window distinct from 20."""
    r = close.pct_change()
    mom = close / close.shift(60) - 1.0
    dd = r[r < 0].rolling(n, min_periods=n // 2).std()
    return mom / dd.replace(0, np.nan)


def main():
    closes = load_closes()

    cands = {}
    cands["semi_dev_ratio_20"] = {a: semi_dev_ratio(s, 20) for a, s in closes.items()}
    cands["long_short_vol_diff_10_60"] = {a: long_short_vol_diff(s, 10, 60) for a, s in closes.items()}
    cands["monthly_seasonal_1"] = {a: monthly_seasonal(s, 1) for a, s in closes.items()}
    cands["rolling_skew3_30"] = {a: rolling_skew3(s, 30) for a, s in closes.items()}
    cands["kayc_eff_60"] = {a: kayc_eff(s, 60) for a, s in closes.items()}
    cands["sortino_60"] = {a: downs_dev_ratio(s, 60) for a, s in closes.items()}

    for name, vals in cands.items():
        try:
            res = evaluate(closes, vals, name, horizon=10)
            print(f"RESULT {name}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
                  f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}")
        except Exception as e:
            print(f"ERROR {name}: {e}")


if __name__ == "__main__":
    main()

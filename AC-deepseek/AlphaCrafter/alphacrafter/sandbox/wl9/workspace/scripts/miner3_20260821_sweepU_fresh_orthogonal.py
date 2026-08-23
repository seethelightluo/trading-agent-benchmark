"""miner_3 (2026-08-21): Sweep U - fresh orthogonal dimensions.

Existing library: momentum (mom_10, mom_120, vixreg), volatility (bb_width, vol_z,
kaufman, rng_pos, skew/kurt, vix_beta_cond, beta_VIX, cny_beta, dxy_corr), streak.

Probe genuinely new dimensions with likely low library correlation, applied
cross-sectionally over the 15-asset universe:
  - gap_reversion_20  : 20d mean of overnight gap (open/prev_close - 1)
  - us10y_trend_60    : US10Y 60d yield change as a risk-regime signal (all assets)
  - cny10y_minus_us10y: yield-curve/spread differential change
  - up_down_20        : upside capture / downside capture asymmetry over 20d
  - range_pos_z20     : z-score of 20d range position (where in the 20d range price sits)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes


def load_ohlc():
    out = {}
    for a in ASSETS:
        import pathlib
        f = pathlib.Path(f"../persistent/stock_data/{a}.csv")
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out


def main():
    closes = load_closes()
    print("assets:", len(closes))
    ohlc = load_ohlc()

    # gap reversion
    gap = {a: (ohlc[a]["open"] / closes[a].shift(1) - 1.0) for a in closes}
    cand = {
        "gap_rev_20": {a: gap[a].rolling(20).mean() for a in closes},
        "gap_rev_5": {a: gap[a].rolling(5).mean() for a in closes},
    }

    # up/down capture asymmetry over 20d
    ud = {}
    for a in closes:
        r = closes[a].pct_change()
        up = r.clip(lower=0)
        dn = (-r).clip(lower=0)
        upm = up.rolling(20).mean()
        dnm = dn.rolling(20).mean()
        ud[a] = (upm - dnm) / (upm + dnm).replace(0, np.nan)
    cand["up_down_20"] = ud

    # range position z-score over 20d
    rpz = {}
    for a in closes:
        hi = ohlc[a]["high"].rolling(20).max()
        lo = ohlc[a]["low"].rolling(20).min()
        rng = (hi - lo).replace(0, np.nan)
        pos = (closes[a] - lo) / rng
        rpz[a] = (pos - pos.rolling(60).mean()) / pos.rolling(60).std()
    cand["range_pos_z20_60"] = rpz

    # US10Y yield-trend as joint risk regime (broadcast same signal scaled per asset by sign)
    us10y = closes["US10Y"]
    utrend = us10y / us10y.shift(60) - 1.0
    utsign = pd.Series(np.where(utrend.shift(5).notna(), np.where(utrend.shift(5)>0, 1.0, -1.0), np.nan), index=utrend.index)
    cand["us10y_trend_60_sign"] = {a: utsign for a in closes}

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()


if __name__ == "__main__":
    main()
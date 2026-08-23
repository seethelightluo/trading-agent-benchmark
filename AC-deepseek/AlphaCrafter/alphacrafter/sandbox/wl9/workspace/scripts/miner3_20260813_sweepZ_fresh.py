"""miner_3 (2026-08-13): Sweep Z - fresh orthogonal & regime factors.

Explore novel lower-correlation candidate families vs existing library:
  - streak_3d        : 3-day directional streak (count of consecutive same-direction daily moves)
  - gap_thrust_5     : 5d close<->open gap accumulation / open momentum
  - wick_imbalance_10: upper vs lower wick ratio (intraday rejection proxy)
  - vol_trend_ratio  : (realized vol now) vs (realized vol >60d ago) - volatility persistence
  - range_symm_20    : 20d high/low position relative to prior 60d window (regime position)
  - cross_ret_rank   : cross-sectional rank of 5d returns (breadth signal)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import pathlib

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro


def load_ohlc():
    out = {}
    for a in ASSETS:
        f = pathlib.Path(f"../persistent/stock_data/{a}.csv")
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out


def streak(s):
    r = np.sign(s.diff())
    out = pd.Series(np.nan, index=s.index)
    cnt = 0.0
    prev = 0.0
    for i, v in r.items():
        if pd.isna(v):
            cnt = 0.0
            prev = 0.0
            continue
        if v == prev:
            cnt = cnt + v
        else:
            cnt = v
        prev = v
        out[i] = cnt
    return out


def main():
    ohlc = load_ohlc()
    closes = load_closes()
    macro = load_macro()
    ret = {a: closes[a].pct_change() for a in closes}
    cand = {}

    # 3-day streak
    for a in closes:
        cand.setdefault("streak_3d", {})[a] = streak(closes[a])

    # gap thrust: 5d sum of (open/prevclose - 1) normalized by 5d |sum|
    for a in closes:
        o = ohlc[a]["open"]
        pc = closes[a].shift(1)
        gap = o / pc - 1.0
        gap5 = gap.rolling(5).sum()
        gabs = gap.abs().rolling(5).sum()
        cand.setdefault("gap_thrust_5", {})[a] = gap5 / gabs.replace(0, np.nan)

    # wick imbalance 10d: (upper wick sum - lower wick sum) / total range
    for a in closes:
        hi = ohlc[a]["high"]; lo = ohlc[a]["low"]; cl = closes[a]
        upper = (hi - np.maximum(cl, ohlc[a]["open"]))
        lower = (np.minimum(cl, ohlc[a]["open"]) - lo)
        wick = (upper.rolling(10).sum() - lower.rolling(10).sum()) / \
               (hi.rolling(10).max() - lo.rolling(10).min()).replace(0, np.nan)
        cand.setdefault("wick_imb_10", {})[a] = wick

    # vol trend ratio: 20d vol / 20d vol ~60d ago
    for a in closes:
        rv20 = ret[a].rolling(20).std()
        rat = rv20 / rv20.shift(60).replace(0, np.nan)
        cand.setdefault("vol_trend_60", {})[a] = rat

    # range symmetry: current 20d window position vs 60d prior regime (0..1 z-ish)
    for a in closes:
        hi_20 = ohlc[a]["high"].rolling(20).max()
        lo_20 = ohlc[a]["low"].rolling(20).min()
        pos = (closes[a] - lo_20) / (hi_20 - lo_20).replace(0, np.nan)
        mid = (closes[a].rolling(60).mean() - lo_20) / (hi_20 - lo_20).replace(0, np.nan)
        cand.setdefault("range_symm_20", {})[a] = pos - mid

    # cross-sectional rank of 5d returns broadcast (breadth)
    retf = pd.DataFrame(ret)
    ret5 = retf.rolling(5).sum()
    csm = ret5.rank(axis=1, pct=True)
    cand["cross_ret5_rank"] = {a: csm[a] for a in closes}

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()


if __name__ == "__main__":
    main()
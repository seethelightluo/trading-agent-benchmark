"""miner_3 (2026-07-30): Sweep W - orthogonal dimensions with low library correlation.

Recent passing candidates (cs_beta_20, cs_mom_rank20, up_down_20, range_pos_z20_60,
vol_ratio_5_20, price_vol_div) all had max_abs_library_correlation >= 0.5 and would be
evicted for pairwise conflicts. Probe fresh, low-correlation ideas:

  - skew_adj_mom20   : 20d momentum adjusted by 20d skew (sign-conjunction)
  - val_ratio_20     : 20d upside/total range share vs downside (capture ratio, monotone)
  - ewm_accel_5_20   : 5d ewm momentum minus 20d ewm momentum (momentum acceleration)
  - vol_disp_20      : dispersion/std of per-day range over 20d (vol-of-range)
  - half_life_streak : persistence-weighted streak (exponential decay of streak length)
  - cross_ts_rank10  : 10d time-series rank of daily return within rolling window
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes


def load_ohlc():
    import pathlib
    out = {}
    for a in ASSETS:
        f = pathlib.Path(f"../persistent/stock_data/{a}.csv")
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out


def main():
    closes = load_closes()
    print("assets:", len(closes))
    ohlc = load_ohlc()

    cand = {}

    # skew-adjusted momentum: sign(mom20)*abs(mom20)*[1+skew20] style - price*skew
    for a in closes:
        mom = closes[a] / closes[a].shift(20) - 1.0
        sk = closes[a].pct_change().rolling(20).skew()
        cand.setdefault("mom_skew_prod_20", {})[a] = mom * sk

    # upside/total-range share over 20d
    for a in closes:
        r = closes[a].pct_change()
        up = r.clip(lower=0).rolling(20).mean()
        dn = (-r).clip(lower=0).rolling(20).mean()
        rng = (up + dn).replace(0, np.nan)
        cand.setdefault("up_capture_20", {})[a] = up / rng

    # momentum acceleration: 5d ewm - 20d ewm
    for a in closes:
        em5 = closes[a].ewm(span=5).mean()
        em20 = closes[a].ewm(span=20).mean()
        cand.setdefault("ewm_accel_5_20", {})[a] = (closes[a] / em5 - 1.0) - (closes[a] / em20 - 1.0)

    # vol-of-range: std of daily (high-low)/close over 20d
    for a in closes:
        daily_range = (ohlc[a]["high"] - ohlc[a]["low"]) / closes[a].replace(0, np.nan)
        cand.setdefault("vol_of_range_20", {})[a] = daily_range.rolling(20).std()

    # persistence-weighted streak (running sign-count, geometrically decayed)
    for a in closes:
        s = np.sign(closes[a].pct_change())
        out = pd.Series(np.nan, index=closes[a].index)
        acc = 0.0
        prev = 0.0
        for i in range(len(s)):
            v = s.iloc[i]
            if np.isnan(v):
                acc = 0.0
                continue
            if v == prev and prev != 0:
                acc = acc * 0.9 + v
            else:
                acc = v
            prev = v
            out.iloc[i] = acc
        cand.setdefault("persist_streak_09", {})[a] = out

    # 10d time-series rank of daily return (recent relative to own past)
    for a in closes:
        r = closes[a].pct_change()
        trank = r.rolling(20).rank(pct=True)
        cand.setdefault("ts_rank_20", {})[a] = trank

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()


if __name__ == "__main__":
    main()

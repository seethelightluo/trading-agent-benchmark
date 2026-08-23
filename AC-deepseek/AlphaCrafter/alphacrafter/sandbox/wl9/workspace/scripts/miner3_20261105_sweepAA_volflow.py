"""miner_3 (2026-11-05): Sweep AA - volume-flow / liquidity dynamics.

Library already covers momentum, vol level/z, beta (VIX/DXY/CNY), range position,
skew/kurt, kaufman efficiency, days_since_high, streak, corr-change, VIX-regime mom.

Genuinely NEW dimensions probed here (volume is currently only used in vol_z_20d as a
level z-score; no volume-price flow signal captured):
  - vwap_drift_20  : 20d close vs VWAP drift (price trend relative to avg traded price)
  - pv_div_20      : 20d return per unit 20d volume z (price move per flow impulse), z-scored
  - vol_flow_20    : 20d return * sign(volume z) (confirmation: trend + rising volume)
  - amihud_20      : |ret| / volume (illiquidity proxy), 20d mean, z-scored
  - volume_updown_20: has volume rising on up days vs down days (asymmetric flow)
Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; lib corr < 0.5 for persist.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import pathlib

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes


def load_ohlc():
    out = {}
    for a in ASSETS:
        f = pathlib.Path(f"../persistent/stock_data/{a}.csv")
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out


def main():
    closes = load_closes()
    ohlc = load_ohlc()
    print("assets:", len(closes))

    cand = {}

    # VWAP drift: 20d mean of (close - vwap)/vwap, vwap = cum(price*vol)/cum(vol)
    for a in closes:
        df = ohlc[a]
        hp = df["close"] * df["volume"].where(df["volume"] > 0)
        cum_pv = hp.rolling(20, min_periods=10).sum()
        cum_v = df["volume"].where(df["volume"] > 0).rolling(20, min_periods=10).sum()
        vwap = cum_pv / cum_v.replace(0, np.nan)
        vd = df["close"] / vwap.replace(0, np.nan) - 1.0
        cand.setdefault("vwap_drift_20", {})[a] = vd

    # Price move per unit volume flow (volume-flow efficiency), z-scored cross-time
    for a in closes:
        df = ohlc[a]
        ret = df["close"].pct_change()
        r20 = ret.rolling(20, min_periods=10).sum()
        v20 = df["volume"].where(df["volume"] > 0).rolling(20, min_periods=10).mean()
        pv = r20 / v20.replace(0, np.nan)
        z = (pv - pv.rolling(120, min_periods=60).mean()) / pv.rolling(120, min_periods=60).std().replace(0, np.nan)
        cand.setdefault("pv_div_20", {})[a] = z

    # Return * volume-z (confirmation of trend by rising volume)
    for a in closes:
        df = ohlc[a]
        v = df["volume"].where(df["volume"] > 0)
        vz = (v - v.rolling(60, min_periods=30).mean()) / v.rolling(60, min_periods=30).std().replace(0, np.nan)
        mom20 = df["close"] / df["close"].shift(10) - 1.0
        cand.setdefault("vol_flow_20", {})[a] = mom20 * np.sign(vz)

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()


if __name__ == "__main__":
    main()

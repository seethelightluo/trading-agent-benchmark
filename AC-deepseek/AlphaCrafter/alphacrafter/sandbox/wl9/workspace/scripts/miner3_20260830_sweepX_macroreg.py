"""miner_3 (2026-08-30): Sweep X - macro-regime & orthogonal low-correlation factors.

Goal: find factors passing the IC/ICIR gate that ALSO stay below the 0.5 library
correlation eviction bound. Many price/vol momentum variants correlate >0.5 with the
existing library. Probe macro-driven and cross-asset signals:

  - usd_adj_ret_10  : asset 10d momentum "risk-adjusted" by contemporaneous USD regime
                       (multiply by sign of DXY 10d change shifted) - a USD-neutral tilt
  - rate_down_tail  : assets favored when US10Y falling (joint risk-off regime broadcast)
  - cnyyield_spread : change in (CN10Y - US10Y) spread as bond-market regime signal
  - vix_tail_5      : 5d VIX change as a cross-asset common factor (broadcast, scaled by sign)
  - macro_streak    : cyclical high/low count of asset vs US10Y (asset/rate regime)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro


def main():
    closes = load_closes()
    macro = load_macro()
    print("assets:", len(closes), "macro:", len(macro))

    dn = {a: closes[a].index for a in closes}
    vix = macro["VIX"].reindex(closes["SPX"].index)
    dxy = macro["DXY"].reindex(closes["SPX"].index)
    us10 = closes["US10Y"]
    cn10 = closes["CN10Y"]

    cand = {}

    # USD-regime-flipped momentum: 10d momentum * sign(DXY 10d change shifted 5)
    dr = dxy.pct_change(10)
    usdsign = pd.Series(np.where(dr.shift(5).notna(), np.where(dr.shift(5) > 0, -1.0, 1.0), np.nan), index=dr.index)
    for a in closes:
        mom = closes[a] / closes[a].shift(10) - 1.0
        cand.setdefault("usd_adj_mom10", {})[a] = mom * usdsign

    # VIX 5d change broadcast common factor (same sign across assets)
    v5 = vix.pct_change(5)
    vsign = pd.Series(np.where(v5.shift(5).notna(), np.where(v5.shift(5) > 0, -1.0, 1.0), np.nan), index=v5.index)
    cand["vix_5d_sign"] = {a: vsign for a in closes}

    # CN10Y - US10Y spread change (bond market regime) broadcast
    spread = cn10 - us10
    spr = spread.diff(20)
    ssign = pd.Series(np.where(spr.shift(5).notna(), np.where(spr.shift(5) > 0, 1.0, -1.0), np.nan), index=spr.index)
    cand["cny_spread_20_sign"] = {a: ssign for a in closes}

    # US10Y level z-score as joint risk regime (higher yields -> lower equity-like assets)
    uz = (us10 - us10.rolling(60).mean()) / us10.rolling(60).std()
    uzsign = pd.Series(np.where(uz.shift(5).notna(), np.where(uz.shift(5) > 0, -1.0, 1.0), np.nan), index=uz.index)
    cand["us10y_z60_sign"] = {a: uzsign for a in closes}

    # asset 10d momentum risk-adjusted by its own 10d vol (Sharpe-like; distinct from vol_z)
    for a in closes:
        mom = closes[a] / closes[a].shift(10) - 1.0
        vol = closes[a].pct_change().rolling(10).std()
        cand.setdefault("sharpe_mom10", {})[a] = mom / vol

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()


if __name__ == "__main__":
    main()

"""miner_3 (2026-08-21): Sweep W - genuinely orthogonal dimensions (low library corr focus).

Recent passing candidates (cs_beta, up_down, range_pos_z, vol_ratio_5_20, price_vol_div,
cs_mom_rank) all carried maxcorr > 0.5 vs existing library (rng_pos_20d / vol_z_20d /
beta_VIX_60), so they would likely be evicted by the pairwise gate. Probe dimensions
that are structurally different and likely orthogonal to the momentum/vol/beta families:

  - cnyus_slope_20   : CN10Y - US10Y yield-curve slope change (macro rate differential)
  - cross_disp_20    : cross-sectional dispersion of 20d returns (breadth risk regime)
  - ccl_comm_mom_20  : commodity basket (XAU+COPPER+WTI) 20d momentum broadcast; sign
                        matched per-asset as a commodity-beta style signal
  - ret_vs_vol_20    : 20d return / 20d realized vol (risk-adjusted drift), z-scored
  - hi_lo_eff_20     : 20d high-low efficiency (close vs range), a liquidity/slippage proxy
  - mom_curl_20_60   : momentum curvature: (mom20 - mom60), i.e. trend acceleration
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import pathlib

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()

def load_ohlc():
    out = {}
    for a in ASSETS:
        f = pathlib.Path(f"../persistent/stock_data/{a}.csv")
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        df = df[df["date"] <= "2026-07-29"]
        out[a] = df.set_index("date")
    return out

def main():
    ohlc = load_ohlc()
    ret = {a: closes[a].pct_change() for a in closes}
    cand = {}

    # CN10Y - US10Y slope change (rate differential)
    cn = closes["CN10Y"]; us = closes["US10Y"]
    slope = cn - us
    cand["cnyus_slope_20"] = {a: slope.diff(20) for a in closes}

    # cross-sectional dispersion of 20d returns
    retf = pd.DataFrame(ret)
    ret20 = retf.rolling(20).sum()
    cand["cross_disp_20"] = {a: ret20.std(axis=1) for a in closes}

    # commodity basket momentum, per-asset beta-signed
    wti = closes["WTI"]; xau = closes["XAU"]; cop = closes["COPPER"]
    basket = (wti.pct_change() + xau.pct_change() + cop.pct_change()) / 3.0
    bmom = basket.rolling(20).mean()*20
    cand["comm_mom_20"] = {a: bmom for a in closes}

    # risk-adjusted drift
    for a in closes:
        r = ret[a]
        dr = r.rolling(20).sum()
        rv = r.rolling(20).std()
        cand.setdefault("ret_vs_vol_20", {})[a] = dr / rv.replace(0, np.nan)

    # hi-lo efficiency
    for a in closes:
        hi = ohlc[a]["high"].rolling(20).max()
        lo = ohlc[a]["low"].rolling(20).min()
        hi_lo = (closes[a]-lo)/ (hi-lo).replace(0, np.nan)
        # acceleration: momentum curvature 20 vs 60
        mom20 = closes[a]/closes[a].shift(20)-1.0
        mom60 = closes[a]/closes[a].shift(60)-1.0
        cand.setdefault("hi_lo_eff_20", {})[a] = hi_lo
        cand.setdefault("mom_curl_20_60", {})[a] = mom20 - mom60

    for name, vals in cand.items():
        try:
            evaluate(closes, vals, name, horizon=10)
        except Exception as e:
            print(name, "ERROR:", repr(e))
        print()

if __name__ == "__main__":
    main()

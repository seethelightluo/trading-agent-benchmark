"""miner_3 (2026-08-13): Cycle-2 sweep B - regime-flipped momentum with tight orthogonal focus.

Goal: find low-library-corr momentum tilts that pass gate (absIC>=0.0070 &
absICIR>=0.0840). Try regime-flips that do NOT simply duplicate existing vol_z or beta:
  - vixreg_mom10_gap   : 10d momentum X sign(VIX 5d change) [VIX regime momentum]
  - dxy_mom10_flip     : 10d momentum X sign(DXY 60d change) [USD regime long-horizon]
  - eurusd_mom10       : 10d momentum X sign(EURUSD 20d change) [EUR regime tilt]
  - us10_tail_mom10    : 10d momentum X sign(US10Y 20d change) [rates regime tilt]
  - cn10_tail_mom10    : 10d momentum X sign(CN10Y 20d change) [China rates regime]
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
    idx = closes["SPX"].index
    vix = macro["VIX"].reindex(idx)
    dxy = macro["DXY"].reindex(idx)
    eur = macro["EURUSD"].reindex(idx)
    usdj = macro["USDJPY"].reindex(idx)
    us10 = closes["US10Y"]
    cn10 = closes["CN10Y"]

    def regime_flip(regime_series, lookback, shift=5):
        r = regime_series.pct_change(lookback)
        rr = r.shift(shift)
        return pd.Series(np.where(rr.notna(), np.where(rr > 0, -1.0, 1.0), np.nan), index=rr.index)

    flips = {
        "vix_1d": regime_flip(vix, 1, 5),
        "vix_5d": regime_flip(vix, 5, 5),
        "dxy_5d": regime_flip(dxy, 5, 5),
        "dxy_60d": regime_flip(dxy, 60, 5),
        "eur_20d": regime_flip(eur, 20, 5),
        "usdj_20d": regime_flip(jpy := regime_flip, 0, 5) if False else regime_flip(jpy if 'jpy' in dir() else macro["USDJPY"].reindex(idx), 20, 5),
        "us10_20d": regime_flip(us10, 20, 5),
        "cn10_20d": regime_flip(cn10, 20, 5),
    }
    # fix usdj jpy variable
    flips["usdj_20d"] = regime_flip(macro["USDJPY"].reindex(idx), 20, 5)

    for name, fl in flips.items():
        for a in closes:
            mom = closes[a] / closes[a].shift(10) - 1.0
            # build candidate dict on the fly
        vals = {a: (closes[a] / closes[a].shift(10) - 1.0) * fl for a in closes}
        lab = f"mom10_flip_{name}"
        try:
            evaluate(closes, vals, lab, horizon=10)
        except Exception as e:
            print(lab, "ERROR:", repr(e))
        print()

if __name__ == "__main__":
    main()
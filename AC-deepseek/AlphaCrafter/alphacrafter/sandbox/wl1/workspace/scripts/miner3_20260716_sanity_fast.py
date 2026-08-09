"""Sanity check: fast vectorized IC vs miner1_common reference."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close, factor_panel, ic_analysis
import miner3_fast as F

closes = load_close()
fwd1 = F.fwd_returns(closes, 1)

def vol_nd(nd):
    def f(df):
        return df["close"].pct_change().rolling(nd).std() * np.sqrt(252)
    return f

for name, fn in [("vol_20d", vol_nd(20)), ("vol_60d", vol_nd(60))]:
    panel = factor_panel(closes, fn)
    ref = ic_analysis(panel, closes, fwd_days=1)
    fast = F.fast_ic(panel, fwd1)
    print(f"{name}: ref IC={ref['ic']:+.4f} ICIR={ref['icir']:+.3f} n={ref['n_dates']} | "
          f"fast IC={fast['ic']:+.4f} ICIR={fast['icir']:+.3f} n={fast['n_dates']}")
    assert abs(ref["ic"] - fast["ic"]) < 1e-8, f"IC mismatch {name}"
    assert abs(ref["icir"] - fast["icir"]) < 1e-8, f"ICIR mismatch {name}"
print("SANITY OK")
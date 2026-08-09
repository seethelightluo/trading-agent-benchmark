"""Diagnose where the miner3 volatility script is slow."""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close, factor_panel, ic_analysis

t0 = time.time()
closes = load_close()
print(f"load_close: {time.time()-t0:.2f}s symbols={len(closes)} dates={len(closes['SPX'])}")

t0 = time.time()
def vol_nd(nd):
    def f(df):
        return df["close"].pct_change().rolling(nd).std() * np.sqrt(252)
    return f
panel = factor_panel(closes, vol_nd(20))
print(f"factor_panel vol_20d: {time.time()-t0:.2f}s shape={panel.shape}")

t0 = time.time()
ic1 = ic_analysis(panel, closes, fwd_days=1)
print(f"ic_analysis vol_20d: {time.time()-t0:.2f}s -> {ic1}")
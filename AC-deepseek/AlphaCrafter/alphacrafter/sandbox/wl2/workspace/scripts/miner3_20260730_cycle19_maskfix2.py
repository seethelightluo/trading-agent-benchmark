"""miner_3 2026-07-30: gate-recoverable self-mask via np.tile boolean frame."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import build_panel

prices = build_panel()
panel = pd.DataFrame(prices)
env = {"pd": pd, "np": np, "close": panel, "len": len}

MASK = "pd.DataFrame(np.tile({cond}, (len(close), 1)), index=close.index, columns=close.columns)"
tests = {
    "corr_wti60":  f"close.pct_change().rolling(60, min_periods=15).corr(close['WTI'].pct_change()).where({MASK.format(cond='close.columns != \"WTI\"')})",
    "corr_xau60":  f"close.pct_change().rolling(60, min_periods=15).corr(close['XAU'].pct_change()).where({MASK.format(cond='close.columns != \"XAU\"')})",
    "corr_us10y60": f"close.pct_change().rolling(60, min_periods=15).corr(close['US10Y'].pct_change()).where({MASK.format(cond='close.columns != \"US10Y\"')})",
    "corr_btc60":  f"close.pct_change().rolling(60, min_periods=15).corr(close['BTC'].pct_change()).where({MASK.format(cond='close.columns != \"BTC\"')})",
    "rel_mom20_spx": f"(close.shift(5)/close.shift(25)-1.0).sub(close['SPX'].shift(5)/close['SPX'].shift(25)-1.0, axis=0).where({MASK.format(cond='close.columns != \"SPX\"')})",
    "rel_mom60_spx": f"(close.shift(5)/close.shift(65)-1.0).sub(close['SPX'].shift(5)/close['SPX'].shift(65)-1.0, axis=0).where({MASK.format(cond='close.columns != \"SPX\"')})",
}
for fid, exp in tests.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape
        nv = int(sig.notna().sum().sum()) if ok else 0
        ref = 'WTI' if 'wti' in fid else ('XAU' if 'xau' in fid else ('US10Y' if 'us10y' in fid else ('BTC' if 'btc' in fid else 'SPX')))
        ref_nan = float(sig[ref].notna().mean()) if ok else None
        print(f"{fid:16s} eval={'OK' if ok else 'BAD'} valid={nv:6d} {ref}_nonnan_frac={ref_nan if ref_nan is not None else '-'}")
    except Exception as e:
        print(f"{fid:16s} eval=FAIL {type(e).__name__}: {str(e)[:80]}")

"""miner_3 2026-07-30: quick fix test for self-mask in correlation expressions
and SPX relative-momentum alignment in the gate namespace."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import build_panel

prices = build_panel()
panel = pd.DataFrame(prices)
env = {"pd": pd, "np": np, "close": panel}

tests = {
    # mask via boolean Series aligned to columns, axis=1
    "corr_wti60_v1": "close.pct_change().rolling(60, min_periods=15).corr(close['WTI'].pct_change()).where(pd.Series(close.columns != 'WTI', index=close.columns), axis=1)",
    "corr_wti60_v2": "close.pct_change().rolling(60, min_periods=15).corr(close['WTI'].pct_change()).where(~close.columns.isin(['WTI']), axis=1)",
    "corr_xau60_v1": "close.pct_change().rolling(60, min_periods=15).corr(close['XAU'].pct_change()).where(pd.Series(close.columns != 'XAU', index=close.columns), axis=1)",
    "corr_us10y60_v1": "close.pct_change().rolling(60, min_periods=15).corr(close['US10Y'].pct_change()).where(pd.Series(close.columns != 'US10Y', index=close.columns), axis=1)",
    "corr_btc60_v1": "close.pct_change().rolling(60, min_periods=15).corr(close['BTC'].pct_change()).where(pd.Series(close.columns != 'BTC', index=close.columns), axis=1)",
    "rel_mom20_spx": "(close.shift(5)/close.shift(25)-1.0).sub(close['SPX'].shift(5)/close['SPX'].shift(25)-1.0, axis=0)",
    "rel_mom60_spx": "(close.shift(5)/close.shift(65)-1.0).sub(close['SPX'].shift(5)/close['SPX'].shift(65)-1.0, axis=0)",
}
for fid, exp in tests.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape
        nv = int(sig.notna().sum().sum()) if ok else 0
        # check mask worked: WTI/SPX column should be mostly NaN for corr factors
        col = 'WTI' if 'wti' in fid else ('XAU' if 'xau' in fid else ('US10Y' if 'us10y' in fid else ('BTC' if 'btc' in fid else None)))
        col_nan = float(sig[col].notna().mean()) if col else None
        print(f"{fid:16s} eval={'OK' if ok else 'BAD'} valid={nv:6d} {col}_nonnan_frac={col_nan if col_nan is not None else '-'}")
    except Exception as e:
        print(f"{fid:16s} eval=FAIL {type(e).__name__}: {str(e)[:80]}")

"""miner_2 2035-09-17: probe data + re-validate currently effective library factors."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner2_20350917_common import (WATCH, MACRO, load_prices, load_macro,
                                    forward_returns, summarize, library_signals,
                                    rank_ic_series, get_visible_through)

print('visible_through:', get_visible_through().date())
px = load_prices()
macro = load_macro()
print('px shape:', px.shape, 'last date:', px.index[-1].date())
print('macro shape:', macro.shape, 'last date:', macro.index[-1].date())
print('\nPer-asset last close / valid days (through', px.index[-1].date(), '):')
for c in px.columns:
    s = px[c].dropna()
    print(f'  {c:10s} rows={len(s):5d} last={s.iloc[-1]:12.2f}')

ret = px.pct_change()
fr = forward_returns(px)

print('\n===== REVALIDATION OF CURRENTLY EFFECTIVE LIBRARY FACTORS =====')
lib = library_signals(px, ret, macro)

# re-validate on full period AND recent 2y (approx 500 trading days)
for name, sig in lib.items():
    print(f'\n--- {name} (library, EFFECTIVE) ---')
    summarize(sig, fr, f'{name} FULL', recent=500)
    # recent 2y window
    sig_recent = sig.iloc[-504:]
    if len(sig_recent) > 60:
        summarize(sig_recent, fr, f'{name} LAST-2Y', recent=252)

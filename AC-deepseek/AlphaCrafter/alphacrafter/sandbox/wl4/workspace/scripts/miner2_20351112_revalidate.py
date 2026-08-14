"""miner_2 2035-11-12: re-validate currently effective library factors through visible 2035-11-09."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner2_20350917_common import (WATCH, MACRO, load_prices, load_macro,
                                    forward_returns, summarize, library_signals,
                                    get_visible_through)

print('visible_through:', get_visible_through().date())
px = load_prices()
macro = load_macro()
print('px shape:', px.shape, 'last date:', px.index[-1].date())
print('macro shape:', macro.shape, 'last date:', macro.index[-1].date())

ret = px.pct_change()
fr = forward_returns(px)

print('\n===== REVALIDATION OF CURRENTLY EFFECTIVE LIBRARY FACTORS =====')
lib = library_signals(px, ret, macro)

for name, sig in lib.items():
    print(f'\n--- {name} (library, EFFECTIVE) ---')
    summarize(sig, fr, f'{name} FULL', recent=500)
    # recent 2y window (~504 trading days)
    sig_recent = sig.iloc[-504:]
    if len(sig_recent) > 60:
        summarize(sig_recent, fr, f'{name} LAST-2Y', recent=252)

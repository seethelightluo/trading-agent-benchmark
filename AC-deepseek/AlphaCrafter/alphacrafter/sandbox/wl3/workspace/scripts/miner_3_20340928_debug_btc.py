"""Debug btc_beta_60 recent-window coverage oddity (n=60)."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, canonical_grid

prices = load_prices(days=4000)
btc_close = prices['BTC']['close']
print("BTC close: n=%d first=%s last=%s" % (len(btc_close), btc_close.index.min(), btc_close.index.max()))
print("BTC close NaN count:", int(btc_close.isna().sum()))
print("BTC close head:")
print(btc_close.head(8))
print("BTC close tail:")
print(btc_close.tail(8))

# check for NaN runs in BTC close within recent window
recent = btc_close[(btc_close.index >= '2026-07-16')]
print("BTC close recent n=%d NaN=%d" % (len(recent), int(recent.isna().sum())))
nan_runs = recent[recent.isna()]
if len(nan_runs):
    print("NaN dates in recent (first 20):", list(nan_runs.index[:20]))

# per-asset last valid dates
for s, df in prices.items():
    c = df['close']
    valid = c[c.notna()]
    if len(valid):
        print(f"{s:10s} last_valid={valid.index[-1].date()} n={len(valid)}")
    else:
        print(f"{s:10s} NO VALID")

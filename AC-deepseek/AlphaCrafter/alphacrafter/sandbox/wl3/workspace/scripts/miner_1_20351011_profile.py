import time, json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
t0 = time.time()

def load_prices(days=6000):
    out = {}
    for s in WATCHLIST:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is not None and len(df) >= 30:
            df = df.copy(); df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            for c in ['open', 'close', 'high', 'low', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            out[s] = df
    return out

prices = load_prices(6000)
print("load_prices:", round(time.time()-t0, 1), "s;", len(prices), "assets")

from numpy.lib.stride_tricks import sliding_window_view
r = prices['SPX']['close'].pct_change().values
t1 = time.time()
sw = sliding_window_view(r, 20)
vals = np.apply_along_axis(lambda x: np.std(x), 1, sw)
print("sliding apply:", round(time.time()-t1, 2), "s")

# rank_ic speed test
dfc = prices['SPX']['close']
fwd = dfc.shift(-10) / dfc - 1.0
fac = dfc.pct_change().rolling(20).kurt()
common = fac.index.intersection(fwd.index)
t2 = time.time()
ic = {}
for d in common:
    x = fac.loc[d]; y = fwd.loc[d]
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if m.sum() >= 8:
        ic[d] = x[m].rank().corr(y[m].rank())
print("rank_ic loop over", len(common), "dates:", round(time.time()-t2, 2), "s")

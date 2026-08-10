import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, canonical_grid, factor_to_panel, validate_factor

prices = load_prices(days=2500)
grid = canonical_grid(prices)
print("grid type:", type(grid), "n:", len(grid))

def f_month_season_prior(df, s):
    c = df['close']
    y = df.index.year.values
    m = df.index.month.values
    g = df.groupby([df.index.year, df.index.month])['close'].last()
    g = g.reset_index().rename(columns={'level_0': 'year', 'level_1': 'month', 'close': 'last_close'})
    g['prev_close'] = g['last_close'].shift(1)
    g['mret'] = g['last_close'] / g['prev_close'] - 1.0
    g['key'] = g['year'] * 12 + g['month']
    mret_by_key = dict(zip(g['key'], g['mret']))
    vals = np.full(len(df), np.nan)
    keys = y * 12 + m
    for i in range(len(df)):
        yrs = np.arange(2020, y[i])
        ks = yrs * 12 + m[i]
        arr = [mret_by_key.get(k) for k in ks]
        arr = [a for a in arr if a is not None and np.isfinite(a)]
        if len(arr) >= 2:
            vals[i] = float(np.mean(arr))
    return pd.Series(vals, index=df.index)

panel = factor_to_panel(f_month_season_prior, prices)
print("panel shape:", panel.shape, "index type:", type(panel.index))
print("index dtype:", panel.index.dtype)
m = validate_factor('month_season_prior', panel, prices)
print("metrics:", m)

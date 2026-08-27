"""Debug macro beta alignment - fix rolling cov issue."""
from alphacrafter.sim.utils import get_stock_daily_data
import pandas as pd
import numpy as np

watchlist = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=1800)
    if df is not None and len(df) >= 800:
        data[sym] = df.set_index('date')['close']
close_df = pd.DataFrame(data).dropna()
ret_df = close_df.pct_change().dropna()

vix = pd.read_csv('../persistent/index_data/VIX.csv', parse_dates=['date']).set_index('date')['close']
dxy = pd.read_csv('../persistent/index_data/DXY.csv', parse_dates=['date']).set_index('date')['close']
usdcny = pd.read_csv('../persistent/index_data/USDCNY.csv', parse_dates=['date']).set_index('date')['close']

d_vix = vix.pct_change()
d_dxy = dxy.pct_change()
d_usdcny = usdcny.pct_change()

def rolling_beta_corrected(ret_df, macro_ret, wins=60):
    common = ret_df.index.intersection(macro_ret.dropna().index)
    r = ret_df.loc[common]
    m = macro_ret.loc[common]
    # Compute beta per asset using manual rolling regression approach
    beta = pd.DataFrame(index=common, columns=watchlist, dtype=float)
    for asset in watchlist:
        x = m.values
        y = r[asset].values
        beta_vec = np.full_like(y, np.nan, dtype=float)
        for i in range(wins, len(y)):
            xw = x[i-wins:i]
            yw = y[i-wins:i]
            if np.std(xw) < 1e-12 or np.std(yw) < 1e-12:
                continue
            b = np.cov(xw, yw)[0,1] / np.var(xw)
            beta_vec[i] = b
        beta[asset] = beta_vec
    return beta

b_vix = rolling_beta_corrected(ret_df, d_vix, 60)
print(f"beta_VIX_60: shape={b_vix.shape}, valid first col={b_vix['SPX'].notna().sum()}")
print(f"Last date: {b_vix.index[-1].date()}, first: {b_vix.index[0].date()}")
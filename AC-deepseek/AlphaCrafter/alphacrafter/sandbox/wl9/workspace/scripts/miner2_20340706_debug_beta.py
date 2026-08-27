"""Debug macro beta alignment issues."""
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
print(f"Close index: {close_df.index[-1].date()}, {len(close_df)} days")

vix = pd.read_csv('../persistent/index_data/VIX.csv', parse_dates=['date']).set_index('date')['close']
dxy = pd.read_csv('../persistent/index_data/DXY.csv', parse_dates=['date']).set_index('date')['close']
usdcny = pd.read_csv('../persistent/index_data/USDCNY.csv', parse_dates=['date']).set_index('date')['close']
usdjpy = pd.read_csv('../persistent/index_data/USDJPY.csv', parse_dates=['date']).set_index('date')['close']

print(f"VIX index: {vix.index[-1].date()} .. {vix.index[0].date()}, {len(vix)}")
print(f"DXY index: {dxy.index[-1].date()} .. {dxy.index[0].date()}, {len(dxy)}")

d_vix = vix.pct_change()
d_dxy = dxy.pct_change()
d_usdcny = usdcny.pct_change()
d_usdjpy = usdjpy.pct_change()

# Align ret_df dates with macro dates
def rolling_beta_debug(ret_df, macro_ret, wins=60):
    # Align by intersection of index
    common = ret_df.index.intersection(macro_ret.dropna().index)
    r = ret_df.loc[common]
    m = macro_ret.loc[common].rename('M')
    si = pd.concat([r, m], axis=1)
    # Check coverage
    print(f"  Aligned dates: {len(si)}, macro NaN: {si['M'].isna().sum()}")
    print(f"  First macro non-NaN date: {si['M'].first_valid_index()}")
    cov = si[watchlist].rolling(wins).cov(si['M'])
    var = si['M'].rolling(wins).var()
    beta = cov / var
    print(f"  Beta shape: {beta.shape}, NaN count (first 5 cols): {beta.iloc[:,0].isna().sum()}")
    return beta

print("\ntesting beta_VIX_60:")
beta_vix = rolling_beta_debug(ret_df, d_vix, 60)
print("Valid dates:", (~beta_vix.iloc[:,0].isna()).sum())

print("\ntesting beta_DXY_60:")
beta_dxy = rolling_beta_debug(ret_df, d_dxy, 60)
print("Valid dates:", (~beta_dxy.iloc[:,0].isna()).sum())

print("\ntesting beta_USDJPY_60:")
beta_jpy = rolling_beta_debug(ret_df, d_usdjpy, 60)
print("Valid dates:", (~beta_jpy.iloc[:,0].isna()).sum())
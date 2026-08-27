
import sys, os, json, warnings
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

warnings.filterwarnings('ignore')

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load_all_data(days=4000):
    data = {}
    for sym in WATCHLIST:
        df = get_stock_daily_data(symbol=sym, days=days)
        if df is not None:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df['ret'] = df['close'].pct_change()
            data[sym] = df
    for sym in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
        df = get_index_daily_data(symbol=sym, days=days)
        if df is not None:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df['ret'] = df['close'].pct_change()
            data[sym] = df
    return data

print('Loading data...')
data = load_all_data(4000)
print(f'Loaded {len(data)} series.')

common_ix = None
for sym in WATCHLIST:
    if sym in data:
        if common_ix is None:
            common_ix = set(data[sym].index)
        else:
            common_ix = common_ix.intersection(set(data[sym].index))
common_ix = sorted(common_ix)
print(f'Common dates: {len(common_ix)} from {common_ix[0].date()} to {common_ix[-1].date()}')

ret_df = pd.DataFrame({sym: data[sym]['ret'] for sym in WATCHLIST if sym in data}, index=common_ix)
close_df = pd.DataFrame({sym: data[sym]['close'] for sym in WATCHLIST if sym in data}, index=common_ix)
vol_df = pd.DataFrame({sym: data[sym]['volume'] for sym in WATCHLIST if sym in data}, index=common_ix)
print(f'Returns matrix: {ret_df.shape}')

def rank_ic(factor_vals, forward_ret):
    valid = factor_vals.notna() & forward_ret.notna()
    if valid.sum() < 8:
        return np.nan
    f_rank = factor_vals[valid].rank()
    r_rank = forward_ret[valid].rank()
    return f_rank.corr(r_rank)

# === FACTOR COMPUTATIONS ===
def compute_ac1_120d(ret_df, window=120):
    result = pd.DataFrame(index=ret_df.index, columns=ret_df.columns, dtype=float)
    for col in ret_df.columns:
        r = ret_df[col]
        result[col] = r.rolling(window=window, min_periods=60).apply(
            lambda x: pd.Series(x).corr(pd.Series(x).shift(1)) if len(x)>=60 else np.nan, raw=False)
    return result

def compute_bb_width_20d(close_df, window=20):
    result = pd.DataFrame(index=close_df.index, columns=close_df.columns, dtype=float)
    for col in close_df.columns:
        c = close_df[col]
        result[col] = 4 * c.rolling(window).std() / c.rolling(window).mean()
    return result

def compute_beta_vix_60(ret_df, window=60):
    vix_ret = ret_df['VIX'] if 'VIX' in close_df.columns else ret_df.get('VIX', None)
    # Actually VIX is not in ret_df columns, need to add it
    return None

print('Factor computations will be done...')
print('Script loaded OK')

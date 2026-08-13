"""miner2 panel builder 2034-04-21. Builds cross-asset panel up to visible date 2034-04-20."""
import pandas as pd, numpy as np

SYMS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
        'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']
END = '2034-04-20'

def load(sym, folder='stock_data'):
    df = pd.read_csv(f'../persistent/{folder}/{sym}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].set_index('date').sort_index()
    return df

close = pd.DataFrame({s: load(s)['close'] for s in SYMS})
open_ = pd.DataFrame({s: load(s)['open'] for s in SYMS})
high = pd.DataFrame({s: load(s)['high'] for s in SYMS})
low = pd.DataFrame({s: load(s)['low'] for s in SYMS})
vol = pd.DataFrame({s: load(s)['volume'] for s in SYMS})
ret = close.pct_change()

macro = pd.DataFrame({m: load(m, 'index_data')['close'] for m in MACRO})

idx = close.index
macro = macro.reindex(idx).ffill()

panel = {'close': close, 'open': open_, 'high': high, 'low': low,
         'vol': vol, 'ret': ret, 'macro': macro}
panel['close'].to_pickle('scripts/miner2_panel_20340421.pkl')
print('panel saved. shape:', close.shape, 'dates:', idx.min().date(), '->', idx.max().date())
print('rows per symbol non-null close:')
print(close.notna().sum())
print('macro tail:')
print(macro.tail(3))

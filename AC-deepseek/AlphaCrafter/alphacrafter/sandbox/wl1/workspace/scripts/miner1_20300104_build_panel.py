"""miner1 2030-01-04: rebuild shared cross-asset panel through most recent completed day (2030-01-03)."""
import pandas as pd, numpy as np

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
END = pd.Timestamp('2030-01-03')  # visible data cutoff (decision date 2030-01-04)

closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
for s in WATCH:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv', parse_dates=['date'])
    df = df[(df['date'] >= '2020-01-01') & (df['date'] <= END)]
    df = df.set_index('date').sort_index()
    closes[s] = df['close'].astype(float)
    vols[s] = df['volume'].astype(float)
    highs[s] = df['high'].astype(float)
    lows[s] = df['low'].astype(float)
    opens[s] = df['open'].astype(float)

close_px = pd.DataFrame(closes).sort_index()
vol_px = pd.DataFrame(vols).sort_index()
high_px = pd.DataFrame(highs).sort_index()
low_px = pd.DataFrame(lows).sort_index()
open_px = pd.DataFrame(opens).sort_index()
ret = close_px.pct_change()

macro = {}
for s in MACRO:
    m = pd.read_csv(f'../persistent/index_data/{s}.csv', parse_dates=['date'])
    m['date'] = pd.to_datetime(m['date'])
    m = m.set_index('date').sort_index()
    m = m[(m.index >= '2020-01-01') & (m.index <= END)]
    macro[s] = m['close'].astype(float)
macro_px = pd.DataFrame(macro).sort_index()

panel = {
    'close': close_px, 'open': open_px, 'high': high_px, 'low': low_px,
    'vol': vol_px, 'ret': ret, 'macro': macro_px,
}
with open('scripts/panel_cache_20300104.pkl', 'wb') as f:
    pd.to_pickle(panel, f)

print("close_px shape:", close_px.shape, close_px.index.min().date(), "->", close_px.index.max().date())
print("macro_px shape:", macro_px.shape, macro_px.index.min().date(), "->", macro_px.index.max().date())
valid_cnt = close_px.notna().sum(axis=1)
print("dates with >=8 valid:", int((valid_cnt >= 8).sum()))
print("last row close:")
print(close_px.iloc[-1].round(2))
print("macro last row:")
print(macro_px.iloc[-1].round(4))

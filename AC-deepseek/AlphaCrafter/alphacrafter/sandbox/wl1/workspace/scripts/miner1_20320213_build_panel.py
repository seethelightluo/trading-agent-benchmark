import pandas as pd, numpy as np, json, os, pickle
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
END = '2032-02-12'  # previous completed trading day before 2032-02-13

closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=6000)
    if df is None:
        df = get_index_daily_data(symbol=s, days=6000)
    if df is None:
        print("NO DATA", s); continue
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].set_index('date').sort_index()
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
    p = f'../persistent/index_data/{s}.csv'
    m = pd.read_csv(p)
    m['date'] = pd.to_datetime(m['date'])
    m = m[m['date'] <= END].set_index('date').sort_index()
    macro[s] = m['close'].astype(float)
macro_px = pd.DataFrame(macro).sort_index()

panel = {
    'close': close_px, 'open': open_px, 'high': high_px, 'low': low_px,
    'vol': vol_px, 'ret': ret, 'macro': macro_px,
}
with open('scripts/panel_cache_2032.pkl', 'wb') as f:
    pickle.dump(panel, f)

print("close_px shape:", close_px.shape, close_px.index.min().date(), "->", close_px.index.max().date())
print("macro_px shape:", macro_px.shape)
valid_cnt = close_px.loc[close_px.index >= '2021-01-04'].notna().sum(axis=1)
print("dates >=2021-01-04:", int((close_px.index >= '2021-01-04').sum()))
print("dates with >=8 valid:", int((valid_cnt >= 8).sum()))
print("dates with >=12 valid:", int((valid_cnt >= 12).sum()))
print("per-asset rows:")
print(close_px.notna().sum())
print("last 5 close dates per asset (min/max):")
print("max date per asset:")
print(close_px.apply(lambda c: c.dropna().index.max()))
# macro coverage
print("macro last dates:")
print(macro_px.apply(lambda c: c.dropna().index.max()))
print("VIX non-null count:", macro_px['VIX'].notna().sum(), "last:", macro_px['VIX'].dropna().index.max())

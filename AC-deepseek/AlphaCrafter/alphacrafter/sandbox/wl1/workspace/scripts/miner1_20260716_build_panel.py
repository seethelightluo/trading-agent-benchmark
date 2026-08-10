"""miner1 2026-07-16: build common cross-asset panel (through 2026-07-15)."""
import pandas as pd, numpy as np, json, os
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
END = '2026-07-15'

closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=6000)
    if df is None:
        df = get_index_daily_data(symbol=s, days=6000)
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

# macro signals
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
with open('scripts/panel_cache.pkl', 'wb') as f:
    pd.to_pickle(panel, f)

print("close_px shape:", close_px.shape, close_px.index.min().date(), "->", close_px.index.max().date())
print("macro_px shape:", macro_px.shape)
# common weekday grid from 2021-01-04
wk = close_px.index[close_px.index >= '2021-01-04']
common = close_px.dropna(how='any').index
common = common[(common >= '2021-01-04')]
print("dates >= 2021-01-04:", len(wk), " all-15 common dates:", len(common))
# dates with >=8 valid
valid_cnt = close_px.loc[close_px.index >= '2021-01-04'].notna().sum(axis=1)
print("dates with >=8 valid:", int((valid_cnt >= 8).sum()))
print("dates with >=12 valid:", int((valid_cnt >= 12).sum()))
print("valid_cnt quantiles:", valid_cnt.quantile([0.1, 0.5, 0.9]).round(2).to_dict())
# coverage per asset
print("\nper-asset coverage (>=2021-01-04):")
print(close_px.loc[close_px.index >= '2021-01-04'].notna().sum())

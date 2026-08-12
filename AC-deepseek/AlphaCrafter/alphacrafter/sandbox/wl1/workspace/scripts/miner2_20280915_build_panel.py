"""miner_2 2028-09-15: rebuild shared cross-asset panel through most recent completed day."""
import pandas as pd, numpy as np, pickle
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=7000)
    if df is None:
        df = get_index_daily_data(symbol=s, days=7000)
    if df is None:
        print("MISSING", s); continue
    df['date'] = pd.to_datetime(df['date'])
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
    p = f'../persistent/index_data/{s}.csv'
    m = pd.read_csv(p)
    m['date'] = pd.to_datetime(m['date'])
    m = m.set_index('date').sort_index()
    macro[s] = m['close'].astype(float)
macro_px = pd.DataFrame(macro).sort_index()

panel = dict(close=close_px, open=open_px, high=high_px, low=low_px, vol=vol_px,
             ret=ret, macro=macro_px)
with open("scripts/panel_cache.pkl", "wb") as fh:
    pickle.dump(panel, fh)
print("panel dates:", close_px.index.min().date(), "->", close_px.index.max().date())
print("close shape:", close_px.shape, "macro shape:", macro_px.shape)
print("last close row:")
print(close_px.tail(2).round(4).T)

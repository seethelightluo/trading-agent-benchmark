"""miner1 2031-03-28: rebuild shared cross-asset panel through 2031-03-27 (last completed day)."""
import pandas as pd, numpy as np
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
        print("MISSING", s)
        continue
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
    m = m[m.index <= close_px.index.max()]
    macro[s] = m['close'].astype(float)
macro_px = pd.DataFrame(macro).sort_index()

panel = {
    'close': close_px, 'open': open_px, 'high': high_px, 'low': low_px,
    'vol': vol_px, 'ret': ret, 'macro': macro_px,
}
with open('scripts/panel_cache_20310328.pkl', 'wb') as f:
    pd.to_pickle(panel, f)

print("close_px shape:", close_px.shape, close_px.index.min().date(), "->", close_px.index.max().date())
print("macro_px shape:", macro_px.shape, macro_px.index.min().date(), "->", macro_px.index.max().date())
print("last dates:", list(close_px.index[-5:].astype(str)))
print("coverage last 60d:", close_px.notna().sum().to_dict())
r = close_px.pct_change().tail(60)
print("\nzero-return series (flat artifact) last 60d:")
flat = (r.abs() < 1e-12).sum()
print(flat[flat > 30].to_dict())
print("\nlast 5 mean returns:")
print(r.tail(5).mean(axis=1).round(4).to_dict())
print("\nlast 5 close (all assets):")
print(close_px.tail(5).round(2).to_string())

"""miner1 2033-03-11: rebuild shared cross-asset panel through most recent completed day (<= 2033-03-10)."""
import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

CUTOFF = pd.Timestamp('2033-03-10')
WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=8000)
    if df is None:
        df = get_index_daily_data(symbol=s, days=8000)
    if df is None:
        print("MISSING", s)
        continue
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df = df[df.index <= CUTOFF]
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
    m = m[m.index <= CUTOFF]
    macro[s] = m['close'].astype(float)
macro_px = pd.DataFrame(macro).sort_index()

panel = {
    'close': close_px, 'open': open_px, 'high': high_px, 'low': low_px,
    'vol': vol_px, 'ret': ret, 'macro': macro_px,
}
with open('scripts/panel_cache_20330311.pkl', 'wb') as f:
    pd.to_pickle(panel, f)

print("close_px shape:", close_px.shape, close_px.index.min().date(), "->", close_px.index.max().date())
print("macro last close:", macro_px.tail(1).round(2).to_dict())
print("last dates:", list(close_px.index[-5:].astype(str)))
r = close_px.pct_change().tail(60)
flat = (r.abs() < 1e-12).sum()
print("flat-artifact names (last 60d):", flat[flat > 30].to_dict())
print("VIX last 10:", macro_px['VIX'].tail(10).round(2).to_dict())
print("US10Y last 5:", close_px['US10Y'].tail(5).round(4).to_dict())
print("WTI last 5:", close_px['WTI'].tail(5).round(2).to_dict())

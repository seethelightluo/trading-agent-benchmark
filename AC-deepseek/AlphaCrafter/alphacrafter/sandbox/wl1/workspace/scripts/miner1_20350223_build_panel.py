"""miner_1 2035-02-23: rebuild panel cache through 2035-02-22 (last completed trading day)."""
import numpy as np
import pandas as pd
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
    m = pd.read_csv(f'../persistent/index_data/{s}.csv')
    m['date'] = pd.to_datetime(m['date'])
    m = m.set_index('date').sort_index()
    m = m[m.index <= close_px.index.max()]
    macro[s] = m['close'].astype(float)
macro_px = pd.DataFrame(macro).sort_index()

panel = {
    'close': close_px, 'open': open_px, 'high': high_px, 'low': low_px,
    'vol': vol_px, 'ret': ret, 'macro': macro_px,
}
with open('scripts/panel_cache_20350223.pkl', 'wb') as f:
    pd.to_pickle(panel, f)

print("close_px shape:", close_px.shape, close_px.index.min().date(), "->", close_px.index.max().date())
print("macro_px shape:", macro_px.shape, macro_px.index.min().date(), "->", macro_px.index.max().date())
print("last close dates:", list(close_px.index[-6:].astype(str)))
r = close_px.pct_change()
print("\nzero-return series (flat artifact) last 250d:")
flat = (r.abs() < 1e-12).sum()
print(flat[flat > 30].to_dict())
print("\nVIX last 6:", macro_px['VIX'].tail(6).round(2).to_dict())
print("20d mean daily ret @last:", round(float(r.tail(20).mean().mean()), 5))
print("20d ann vol mean @last:", round(float(r.tail(20).std().mean()*np.sqrt(252)), 2))
print("20d cross-sectional dispersion (mean daily std):", round(float(r.tail(20).std(axis=1).mean()), 5))
print("\n20d asset mean daily ret @last:")
print(r.tail(20).mean().sort_values(ascending=False).round(5).to_dict())
print("\n60d asset mean daily ret @last:")
print(r.tail(60).mean().sort_values(ascending=False).round(5).to_dict())
print("\nvolume null/zero share per asset (last 250d):")
vz = (vol_px.tail(250) <= 0).mean()
print(vz.round(3).to_dict())

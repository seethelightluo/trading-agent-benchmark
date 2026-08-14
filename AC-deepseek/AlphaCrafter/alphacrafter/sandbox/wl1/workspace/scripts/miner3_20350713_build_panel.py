"""miner_3 2035-07-13: rebuild panel cache through the last completed trading day (2035-07-12)."""
import numpy as np
import pandas as pd

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
VIS = pd.Timestamp('2035-07-12')

closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
for s in WATCH:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VIS].set_index('date').sort_index()
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
    m = m[m.index <= VIS]
    macro[s] = m['close'].astype(float)
macro_px = pd.DataFrame(macro).sort_index()

panel = {
    'close': close_px, 'open': open_px, 'high': high_px, 'low': low_px,
    'vol': vol_px, 'ret': ret, 'macro': macro_px,
}
with open('scripts/panel_cache_20350712.pkl', 'wb') as f:
    pd.to_pickle(panel, f)

print("close_px shape:", close_px.shape, close_px.index.min().date(), "->", close_px.index.max().date())
print("macro_px shape:", macro_px.shape, macro_px.index.min().date(), "->", macro_px.index.max().date())
print("last close dates:", list(close_px.index[-8:].astype(str)))
r = close_px.pct_change()
print("\nzero-return series (flat artifact) last 120d:")
flat = (r.abs() < 1e-12).sum()
print(flat[flat > 30].to_dict())
print("\nVIX last 5:", macro_px['VIX'].tail(5).round(2).to_dict())
print("20d mean daily ret @last:", round(float(r.tail(20).mean().mean()), 5))
print("20d ann vol mean @last:", round(float(r.tail(20).std().mean()*np.sqrt(252)), 2))
print("20d cross-sectional dispersion (mean daily std):", round(float(r.tail(20).std(axis=1).mean()), 5))
print("\n20d asset mean daily ret @last:")
print(r.tail(20).mean().sort_values(ascending=False).round(5).to_dict())
print("\n60d asset mean daily ret @last:")
print(r.tail(60).mean().sort_values(ascending=False).round(5).to_dict())

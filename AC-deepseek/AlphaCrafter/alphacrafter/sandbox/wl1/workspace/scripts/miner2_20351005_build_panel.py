"""miner_2 2035-10-05: rebuild panel cache through the last completed trading day (2035-10-04)."""
import numpy as np
import pandas as pd

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
VIS = pd.Timestamp('2035-10-04')

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
with open('scripts/panel_cache_20351004.pkl', 'wb') as f:
    pd.to_pickle(panel, f)

print("close_px shape:", close_px.shape, close_px.index.min().date(), "->", close_px.index.max().date())
print("macro_px shape:", macro_px.shape, macro_px.index.min().date(), "->", macro_px.index.max().date())
print("last close dates:", list(close_px.index[-6:].astype(str)))
r = close_px.pct_change()
print("\nzero-return series (flat artifact) last 120d:")
flat = (r.abs() < 1e-12).sum()
print(flat[flat > 30].to_dict())
print("\nVIX last 6:", macro_px['VIX'].tail(6).round(2).to_dict())
print("DXY last 3:", macro_px['DXY'].tail(3).round(2).to_dict())
print("USDJPY last 3:", macro_px['USDJPY'].tail(3).round(2).to_dict())
print("EURUSD last 3:", macro_px['EURUSD'].tail(3).round(2).to_dict())
print("20d mean daily ret @last:", round(float(r.tail(20).mean().mean()), 5))
print("20d ann vol mean @last:", round(float(r.tail(20).std().mean()*np.sqrt(252)), 2))
print("20d cross-sectional dispersion (mean daily std):", round(float(r.tail(20).std(axis=1).mean()), 5))
ma20 = close_px.rolling(20).mean()
ma60 = close_px.rolling(60).mean()
print("breadth above MA20:", int((close_px.tail(20) > ma20.tail(20)).iloc[-1].sum()), "/15")
print("breadth above MA60:", int((close_px.tail(60) > ma60.tail(60)).iloc[-1].sum()), "/15")
print("\n20d returns per asset:")
print((close_px.pct_change(20).tail(1).T.round(3)).to_string())
print("\n60d returns per asset:")
print((close_px.pct_change(60).tail(1).T.round(3)).to_string())

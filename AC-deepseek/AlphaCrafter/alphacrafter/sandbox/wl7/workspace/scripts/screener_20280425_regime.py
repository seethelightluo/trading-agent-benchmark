"""Screener regime assessment - data through 2028-04-24 (visible), decision 2028-04-25."""
import pandas as pd, numpy as np, json, os

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = ['DXY','USDCNY','USDJPY','EURUSD','VIX']

closes = {}
for s in WATCH:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = df['date'].astype(str)
    df = df[df['date'] <= '2028-04-24'].reset_index(drop=True)
    closes[s] = df.set_index('date')['close']

px = pd.DataFrame(closes).dropna(how='all')
px = px.sort_index()
print("rows:", len(px), "last:", px.index[-1])

def ret(s, n):
    return px[s].iloc[-1] / px[s].iloc[-1-n] - 1 if len(px) > n else np.nan

print("\n=== 15-asset 20d/10d/5d returns, MA20/MA60 position (data thru 2028-04-24) ===")
rows = []
for s in WATCH:
    r20, r10, r5 = ret(s,20), ret(s,10), ret(s,5)
    ma20 = px[s].rolling(20).mean().iloc[-1]
    ma60 = px[s].rolling(60).mean().iloc[-1]
    last = px[s].iloc[-1]
    above20 = last > ma20
    above60 = last > ma60
    rows.append((s, r20, r10, r5, above20, above60, last))
    print(f"{s:10s} r20 {r20*100:7.2f}%  r10 {r10*100:7.2f}%  r5 {r5*100:7.2f}%  >MA20 {above20}  >MA60 {above60}")

# Market breadth
ab20 = sum(1 for r in rows if r[4])
ab60 = sum(1 for r in rows if r[5])
print(f"\nBreadth: {ab20}/15 above MA20, {ab60}/15 above MA60")

# EW market return
ew = px.pct_change().mean(axis=1)
mkt_20 = (1+ew.iloc[-20:]).prod() - 1
mkt_10 = (1+ew.iloc[-10:]).prod() - 1
mkt_5 = (1+ew.iloc[-5:]).prod() - 1
print(f"\nEW mkt: mkt_20 {mkt_20*100:.2f}%  mkt_10 {mkt_10*100:.2f}%  mkt_5 {mkt_5*100:.2f}%")

# Dispersion: cross-sectional std of 20d returns and max-min
r20s = np.array([ret(s,20) for s in WATCH])
disp_std = np.nanstd(r20s)
disp_range = np.nanmax(r20s) - np.nanmin(r20s)
print(f"Dispersion 20d: std {disp_std*100:.2f}pp, max-min {disp_range*100:.2f}pp")

# Mean pairwise correlation of 20d returns (using last 60 days)
r60 = px.pct_change().iloc[-60:]
c = r60.corr()
mask = np.triu(np.ones(c.shape), k=1).astype(bool)
mean_corr = c.values[mask]
mean_corr = mean_corr[np.isfinite(mean_corr)]
print(f"Mean pairwise corr (60d): {np.mean(mean_corr):.3f}")

# VIX and DXY
for s in OBS:
    df = pd.read_csv(f'../persistent/index_data/{s}.csv')
    df['date'] = df['date'].astype(str)
    df = df[df['date'] <= '2028-04-24'].reset_index(drop=True)
    last = df['close'].iloc[-1]
    r5 = last/df['close'].iloc[-6]-1 if len(df)>6 else np.nan
    r20 = last/df['close'].iloc[-21]-1 if len(df)>21 else np.nan
    r60 = last/df['close'].iloc[-61]-1 if len(df)>61 else np.nan
    print(f"{s}: last {last:.2f}  r5 {r5*100:+.2f}%  r20 {r20*100:+.2f}%  r60 {r60*100:+.2f}%")

# Volatility: EW 20d realized vol annualized (median across assets)
vols = px.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
print(f"\n20d realized vol by asset (ann.):"); 
for s in WATCH:
    print(f"  {s:10s} {vols[s]*100:.1f}%")
print(f"median vol: {np.nanmedian(vols)*100:.1f}%")

# max drawdown over last 60d for EW
ew_cum = (1+ew).cumprod().iloc[-60:]
dd = (ew_cum / ew_cum.cummax() - 1).min()
print(f"EW 60d max drawdown: {dd*100:.2f}%")

# Frozen feed check: last 20 days flat?
print("\nFrozen feed check (std of last 20 closes = 0):")
for s in WATCH:
    seg = px[s].iloc[-20:]
    if seg.std() == 0 or np.isnan(seg.std()):
        print(f"  {s}: FROZEN (std={seg.std()})")

"""Screener regime assessment as of 2031-08-07 (visible through 2031-08-06)."""
import pandas as pd
import numpy as np

VIS = '2031-08-06'
ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']

def load(path, name):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VIS].reset_index(drop=True)
    df = df.rename(columns={'close': name})
    return df[['date', name]]

px = None
for a in ASSETS + OBS:
    path = f'../persistent/stock_data/{a}.csv' if a in ASSETS else f'../persistent/index_data/{a}.csv'
    d = load(path, a)
    px = d if px is None else px.merge(d, on='date', how='outer')
px = px.sort_values('date').reset_index(drop=True)

print("Last date:", px['date'].iloc[-1], " rows:", len(px))

rets = px.set_index('date')[ASSETS].pct_change()
last = px['date'].iloc[-1]

rows = []
for a in ASSETS:
    s = px[a]
    r = rets[a]
    c = s.iloc[-1]
    r10 = (1 + r.tail(10)).prod() - 1
    r20 = (1 + r.tail(20)).prod() - 1
    r60 = (1 + r.tail(60)).prod() - 1
    ma20 = s.rolling(20).mean().iloc[-1]
    ma60 = s.rolling(60).mean().iloc[-1]
    vol20 = r.tail(20).std() * np.sqrt(252)
    vol60 = r.tail(60).std() * np.sqrt(252)
    # consecutive up/down days (last 10)
    sgn = np.sign(r.tail(10).values)
    streak = 0
    for v in sgn[::-1]:
        if v == 0:
            continue
        if streak == 0:
            streak = v
        elif np.sign(v) == np.sign(streak):
            streak += v
        else:
            break
    rows.append(dict(asset=a, close=c, r10=r10*100, r20=r20*100, r60=r60*100,
                     above_ma20=c > ma20, above_ma60=c > ma60,
                     vol20=vol20*100, vol60=vol60*100, streak=streak))
tab = pd.DataFrame(rows).sort_values('r10', ascending=False)
pd.set_option('display.width', 200)
print(tab.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# cross-sectional dispersion of 10d returns
disp10 = rets.tail(10).std(axis=1).mean() * 100
disp20 = rets.tail(20).std(axis=1).mean() * 100
print(f"\nCross-sectional 10d return dispersion (avg daily): {disp10:.3f}%  (20d: {disp20:.3f}%)")
print(f"10d ret range: max {tab.r10.max():.2f}% ({tab.loc[tab.r10.idxmax(),'asset']}), min {tab.r10.min():.2f}% ({tab.loc[tab.r10.idxmin(),'asset']})")

# macro obs
print("\n--- Observation-only signals ---")
for a in OBS:
    s = px[a]
    r = s.pct_change()
    r10 = (1 + r.tail(10)).prod() - 1
    r20 = (1 + r.tail(20)).prod() - 1
    r60 = (1 + r.tail(60)).prod() - 1
    ma20 = s.rolling(20).mean().iloc[-1]
    ma60 = s.rolling(60).mean().iloc[-1]
    print(f"{a:8s} close {s.iloc[-1]:9.2f}  10d {r10*100:+7.2f}%  20d {r20*100:+7.2f}%  60d {r60*100:+7.2f}%  aboveMA20 {s.iloc[-1]>ma20}  aboveMA60 {s.iloc[-1]>ma60}")

# VIX level & regime
vix = px['VIX']
print(f"\nVIX last {vix.iloc[-1]:.2f}, 20d mean {vix.tail(20).mean():.2f}, 60d mean {vix.tail(60).mean():.2f}, min/max 60d {vix.tail(60).min():.1f}/{vix.tail(60).max():.1f}")
print(f"VIX 10d change {(vix.iloc[-1]/vix.iloc[-11]-1)*100:+.1f}%")

# correlation regime: avg pairwise corr of 20d returns across tradable
corr20 = rets.tail(20).corr()
mask = np.triu(np.ones_like(corr20, dtype=bool), k=1)
print(f"\nAvg pairwise 20d correlation (tradable): {corr20.values[mask].mean():.3f}")
corr60 = rets.tail(60).corr()
print(f"Avg pairwise 60d correlation (tradable): {corr60.values[mask].mean():.3f}")

# trend strength proxy: fraction of assets above MA20/MA60
print(f"\nBreadth: assets above MA20: {(tab.above_ma20).sum()}/15, above MA60: {(tab.above_ma60).sum()}/15")

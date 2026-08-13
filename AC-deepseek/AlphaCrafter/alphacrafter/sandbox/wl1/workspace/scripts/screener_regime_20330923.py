import pandas as pd, numpy as np, json

CUR = '2033-09-22'  # visible_through (last completed trading day)
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = {'VIX':'../persistent/index_data/VIX.csv','DXY':'../persistent/index_data/DXY.csv'}

closes = {}
for a in ASSETS:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUR].set_index('date')['close']
    closes[a] = df

px = pd.DataFrame(closes).sort_index()
px = px.dropna(how='all')
print('rows through', CUR, ':', len(px))
print('last date:', px.index[-1].date(), '| first date:', px.index[0].date())

ret = px.pct_change()
# 20d / 60d returns
r20 = (px.iloc[-1] / px.iloc[-21] - 1) * 100
r60 = (px.iloc[-1] / px.iloc[-61] - 1) * 100
ma20 = px.iloc[-1] > px.rolling(20).mean().iloc[-1]
ma60 = px.iloc[-1] > px.rolling(60).mean().iloc[-1]

# 20d mean daily ret (cross-asset equal-weight)
mean20 = ret.tail(20).mean().mean() * 100
mean60 = ret.tail(60).mean().mean() * 100

# cross-sectional dispersion: std of daily returns across assets
disp20 = ret.tail(20).std(axis=1).mean() * 100
disp60 = ret.tail(60).std(axis=1).mean() * 100

# ann vol per asset (20d)
vol20 = ret.tail(20).std() * np.sqrt(252) * 100
vol60 = ret.tail(60).std() * np.sqrt(252) * 100

print('\n=== 20d / 60d return (%) ===')
for a in ASSETS:
    print(f'{a:10s} r20 {r20[a]:+8.2f}  r60 {r60[a]:+8.2f}  aboveMA20 {ma20[a]}  aboveMA60 {ma60[a]}  vol20 {vol20[a]:5.1f}%')

print('\n=== aggregates (through %s) ===' % CUR)
print(f'mean daily 20d: {mean20:+.4f}% | 60d: {mean60:+.4f}%')
print(f'breadth above MA20: {ma20.sum()}/15, above MA60: {ma60.sum()}/15')
print(f'cross-sectional dispersion 20d: {disp20:.3f}% daily | 60d: {disp60:.3f}% daily')
print(f'mean 20d ann vol: {vol20.mean():.1f}% | median: {vol20.median():.1f}%')

for name, path in OBS.items():
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUR].set_index('date')['close']
    print(f'{name}: last {df.iloc[-1]:.1f} | 20d ago {df.iloc[-21]:.1f} | 60d ago {df.iloc[-61]:.1f} | 20d chg {100*(df.iloc[-1]/df.iloc[-21]-1):+.1f}%')

# momentum leaders/laggards 20d
print('\n20d leaders:', ', '.join(f'{a} {r20[a]:+.1f}' for a in r20.sort_values(ascending=False).index[:5]))
print('20d laggards:', ', '.join(f'{a} {r20[a]:+.1f}' for a in r20.sort_values(ascending=True).index[:5]))
print('60d leaders:', ', '.join(f'{a} {r60[a]:+.1f}' for a in r60.sort_values(ascending=False).index[:5]))
print('60d laggards:', ', '.join(f'{a} {r60[a]:+.1f}' for a in r60.sort_values(ascending=True).index[:5]))

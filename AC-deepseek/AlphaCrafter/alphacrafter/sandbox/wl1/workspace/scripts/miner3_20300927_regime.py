"""miner3 2030-09-27: quick regime snapshot at latest data."""
import pandas as pd, numpy as np

with open('scripts/panel_cache_20300927.pkl', 'rb') as f:
    panel = pd.read_pickle(f)
px = panel['close']; mpx = panel['macro']; ret = panel['ret']

df = px.dropna(how='all')
mdf = mpx.dropna(how='all')
print('=== last date per series ===')
print('tradables last:', df.index.max().date(), '| macro last:', mdf.index.max().date(), '| rows:', len(df))

r5  = (df.iloc[-1] / df.iloc[-6] - 1) * 100
r20 = (df.iloc[-1] / df.iloc[-21] - 1) * 100
r60 = (df.iloc[-1] / df.iloc[-61] - 1) * 100

ma20 = df.rolling(20).mean().iloc[-1]
ma60 = df.rolling(60).mean().iloc[-1]
ma200 = df.rolling(200).mean().iloc[-1]

v = ret.tail(20).std() * np.sqrt(252) * 100
mean20 = ret.tail(20).mean() * 100

out = pd.DataFrame({
    'r5d%': r5.round(2), 'r20d%': r20.round(2), 'r60d%': r60.round(2),
    'ma20_dist%': ((df.iloc[-1]/ma20 - 1)*100).round(2),
    'ma60_dist%': ((df.iloc[-1]/ma60 - 1)*100).round(2),
    'ma200_dist%': ((df.iloc[-1]/ma200 - 1)*100).round(2),
    'vol20_ann%': v.round(1),
    'mean20_daily%': (mean20*100).round(3)
})
print(out.to_string())

print('\n=== macro last 20d ===')
for s in mdf.columns:
    vv = mdf[s].dropna()
    r20m = (vv.iloc[-1]/vv.iloc[-21] - 1)*100 if len(vv) > 21 else np.nan
    r5m = (vv.iloc[-1]/vv.iloc[-6] - 1)*100 if len(vv) > 6 else np.nan
    print(f'{s}: last={vv.iloc[-1]:.2f} r5d={r5m:.2f}% r20d={r20m:.2f}%')

print('\n=== cross-asset ===')
print('mean |r20| (dispersion):', round(np.abs(r20).mean(),2))
print('n above MA20:', int((df.iloc[-1] > ma20).sum()), '/', len(df.columns))
print('n above MA200:', int((df.iloc[-1] > ma200).sum()), '/', len(df.columns))
c = ret.tail(20).corr()
vals = c.values[np.triu_indices_from(c.values, 1)]
print('avg pairwise corr (20d):', round(np.nanmean(vals),2))
vv = mdf['VIX'].dropna()
print('VIX now:', round(vv.iloc[-1],2), '| 20d ago:', round(vv.iloc[-21],2) if len(vv)>21 else 'n/a')
print('VIX pct 20d:', round((vv.iloc[-1]/vv.iloc[-21]-1)*100,1) if len(vv)>21 else 'n/a', '%')

m = ret.tail(40).mean(axis=1)
sig = np.sign(m)
streak = 0
for x in sig[::-1]:
    if x == sig.iloc[-1] and x != 0: streak += 1
    else: break
print('cross-asset mean daily ret sign streak (40d):', streak, 'last sign:', sig.iloc[-1])
slope = (ma20 - ma20.shift(5)).iloc[-1] / ma20.iloc[-1] * 100
print('MA20 5d slope %:', round(slope,3))

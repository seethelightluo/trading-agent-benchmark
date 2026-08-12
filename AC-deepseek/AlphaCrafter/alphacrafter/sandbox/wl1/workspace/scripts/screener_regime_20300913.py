import pandas as pd, numpy as np, os

base = '../persistent/stock_data'
tradables = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
macro = ['DXY','USDCNY','USDJPY','EURUSD','VIX']
CUT = '2030-09-12'

def load(sym, base):
    df = pd.read_csv(os.path.join(base, sym + '.csv'))
    df = df.rename(columns={df.columns[0]: 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    return df['close'] if 'close' in df.columns else df.iloc[:,1]

px = {}
for s in tradables:
    try: px[s] = load(s, base)
    except Exception as e: print('ERR', s, e)
mpx = {}
for s in macro:
    try: mpx[s] = load(s, '../persistent/index_data')
    except Exception as e: print('ERR macro', s, e)

df = pd.DataFrame(px).dropna(how='all')
mdf = pd.DataFrame(mpx).dropna(how='all')
df = df[df.index <= CUT]
mdf = mdf[mdf.index <= CUT]
print('=== last date per series ===')
print('tradables last:', df.index.max().date(), '| macro last:', mdf.index.max().date(), '| rows:', len(df))

r5  = (df.iloc[-1] / df.iloc[-6] - 1) * 100
r20 = (df.iloc[-1] / df.iloc[-21] - 1) * 100
r60 = (df.iloc[-1] / df.iloc[-61] - 1) * 100

ma20 = df.rolling(20).mean().iloc[-1]
ma60 = df.rolling(60).mean().iloc[-1]
ma200 = df.rolling(200).mean().iloc[-1]

ret = df.pct_change()
vol20 = ret.tail(20).std() * np.sqrt(252) * 100
vol60 = ret.tail(60).std() * np.sqrt(252) * 100
mean20 = ret.tail(20).mean() * 100

out = pd.DataFrame({
    'r5d%': r5.round(2), 'r20d%': r20.round(2), 'r60d%': r60.round(2),
    'ma20_dist%': ((df.iloc[-1]/ma20 - 1)*100).round(2),
    'ma60_dist%': ((df.iloc[-1]/ma60 - 1)*100).round(2),
    'ma200_dist%': ((df.iloc[-1]/ma200 - 1)*100).round(2),
    'vol20_ann%': vol20.round(1), 'vol60_ann%': vol60.round(1),
    'mean20_daily%': (mean20*100).round(3)
})
print(out.to_string())

print('\n=== macro last 20d ===')
for s in mdf.columns:
    v = mdf[s].dropna()
    r20m = (v.iloc[-1]/v.iloc[-21] - 1)*100 if len(v) > 21 else np.nan
    r5m = (v.iloc[-1]/v.iloc[-6] - 1)*100 if len(v) > 6 else np.nan
    print(f'{s}: last={v.iloc[-1]:.2f} r5d={r5m:.2f}% r20d={r20m:.2f}%')

print('\n=== cross-asset ===')
print('mean |r20| (dispersion):', round(np.abs(r20).mean(),2))
print('n above MA20:', int((df.iloc[-1] > ma20).sum()), '/', len(df.columns))
print('n above MA200:', int((df.iloc[-1] > ma200).sum()), '/', len(df.columns))
c = ret.tail(20).corr()
vals = c.values[np.triu_indices_from(c.values, 1)]
print('avg pairwise corr (20d):', round(np.nanmean(vals),2))
v = mdf['VIX'].dropna()
print('VIX now:', round(v.iloc[-1],2), '| 20d ago:', round(v.iloc[-21],2) if len(v)>21 else 'n/a', '| 60d ago:', round(v.iloc[-61],2) if len(v)>61 else 'n/a')
print('VIX pct 20d:', round((v.iloc[-1]/v.iloc[-21]-1)*100,1) if len(v)>21 else 'n/a', '%')

m = ret.tail(40).mean(axis=1)
sig = np.sign(m)
streak = 0
for x in sig[::-1]:
    if x == sig.iloc[-1] and x != 0: streak += 1
    else: break
print('cross-asset mean daily ret sign streak (40d):', streak, 'last sign:', sig.iloc[-1])
slope = (ma20 - ma20.shift(5)).iloc[-1] / ma20.iloc[-1] * 100
print('MA20 5d slope %:', round(slope,3))

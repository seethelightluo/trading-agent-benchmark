import pandas as pd, numpy as np, os, json

base = '../persistent/stock_data'
tradables = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
macro = ['DXY','USDCNY','USDJPY','EURUSD','VIX']

def load(sym, base):
    df = pd.read_csv(os.path.join(base, sym + '.csv'), parse_dates=['date'] if 'date' in open(os.path.join(base, sym+'.csv')).readline() else None)
    if 'date' not in df.columns:
        # try first col
        df = df.rename(columns={df.columns[0]: 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    return df['close'] if 'close' in df.columns else df.iloc[:,1]

px = {}
for s in tradables:
    try:
        px[s] = load(s, base)
    except Exception as e:
        print('ERR', s, e)

mpx = {}
for s in macro:
    try:
        mpx[s] = load(s, '../persistent/index_data')
    except Exception as e:
        print('ERR macro', s, e)

df = pd.DataFrame(px).dropna(how='all')
mdf = pd.DataFrame(mpx).dropna(how='all')

end = '2030-08-01'
df = df[df.index <= end]
mdf = mdf[mdf.index <= end]

print('=== last date per series ===')
print(df.index.max().date(), '| macro last:', mdf.index.max().date())
print('n rows:', len(df))

# 20d and 5d returns
r20 = (df.iloc[-1] / df.iloc[-21] - 1) * 100 if len(df) > 21 else np.nan
r5  = (df.iloc[-1] / df.iloc[-6] - 1) * 100 if len(df) > 6 else np.nan
r60 = (df.iloc[-1] / df.iloc[-61] - 1) * 100 if len(df) > 61 else np.nan

# MA20 vs close, MA200
ma20 = df.rolling(20).mean().iloc[-1]
ma60 = df.rolling(60).mean().iloc[-1]
if len(df) > 200:
    ma200 = df.rolling(200).mean().iloc[-1]
else:
    ma200 = pd.Series(np.nan, index=df.columns)

# realized vol 20d ann
ret = df.pct_change()
vol20 = ret.tail(20).std() * np.sqrt(252) * 100
vol60 = ret.tail(60).std() * np.sqrt(252) * 100

# 20d mean daily return
mean20 = ret.tail(20).mean() * 100

out = pd.DataFrame({
    'r5d%': r5.round(2), 'r20d%': r20.round(2), 'r60d%': r60.round(2),
    'ma20_dist%': ((df.iloc[-1]/ma20 - 1)*100).round(2),
    'ma200_dist%': ((df.iloc[-1]/ma200 - 1)*100).round(2),
    'vol20_ann%': vol20.round(1), 'vol60_ann%': vol60.round(1),
    'mean20_daily%': (mean20*100).round(3)
})
print(out.to_string())

print('\n=== macro last 20d ===')
mr = mdf.pct_change()
for s in mdf.columns:
    v = mdf[s]
    r20m = (v.iloc[-1]/v.iloc[-21] - 1)*100 if len(v) > 21 else np.nan
    r5m = (v.iloc[-1]/v.iloc[-6] - 1)*100 if len(v) > 6 else np.nan
    print(f'{s}: last={v.iloc[-1]:.2f} r5d={r5m:.2f}% r20d={r20m:.2f}%')

# cross-asset stats
print('\n=== cross-asset ===')
print('mean |r20| (dispersion):', np.abs(r20).mean().round(2))
print('n above MA20:', int((df.iloc[-1] > ma20).sum()), '/', len(df.columns))
print('n above MA200:', int((df.iloc[-1] > ma200).sum()), '/', len(df.columns))
c = ret.tail(20).corr()
print('avg pairwise corr (20d):', (c.values[np.triu_indices_from(c.values, 1)]).mean().round(2))
# VIX trend
v = mdf['VIX'].dropna()
print('VIX 60d ago:', v.iloc[-61] if len(v)>61 else 'n/a', '-> now:', v.iloc[-1])
print('VIX pct 20d:', ((v.iloc[-1]/v.iloc[-21]-1)*100).round(1), '%')

import pandas as pd, numpy as np, json

VIS = '2035-04-11'  # visible through
assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(f):
    df = pd.read_csv(f'../persistent/stock_data/{f}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= pd.Timestamp(VIS)].reset_index(drop=True)
    return df

closes = {}
rets = {}
for a in assets:
    df = load(a)
    closes[a] = df.set_index('date')['close']
    rets[a] = df.set_index('date')['pct_change']/100.0

px = pd.DataFrame(closes)
rt = pd.DataFrame(rets)
px = px.dropna(how='all'); rt = rt.dropna(how='all')
print('last date in px:', px.index.max().date(), 'n_dates:', len(px))

print('\n=== Recent returns by asset through %s ===' % VIS)
for a in assets:
    s = px[a].dropna()
    if len(s) < 22:
        print(f'{a:<12} insufficient'); continue
    r5  = s.iloc[-1]/s.iloc[-6]-1
    r21 = s.iloc[-1]/s.iloc[-22]-1
    r63 = s.iloc[-1]/s.iloc[-64]-1 if len(s)>64 else np.nan
    hi21 = s.iloc[-21:].max(); lo21 = s.iloc[-21:].min()
    print(f'{a:<12} 5d={r5*100:7.2f}% 21d={r21*100:7.2f}% 63d={r63*100:7.2f}%  | 21d range {lo21:.2f}-{hi21:.2f} last={s.iloc[-1]:.2f}')

spx = px['SPX'].dropna()
ma20 = spx.rolling(20).mean(); ma60 = spx.rolling(60).mean()
print('\n=== SPX trend ===')
print('last close:', spx.iloc[-1], 'ma20:', ma20.iloc[-1], 'ma60:', ma60.iloc[-1])
print('ma20 slope (5d):', (ma20.iloc[-1]/ma20.iloc[-6]-1)*100, '%')
print('above ma20:', spx.iloc[-1]>ma20.iloc[-1], 'above ma60:', spx.iloc[-1]>ma60.iloc[-1])
print('21d ret:', (spx.iloc[-1]/spx.iloc[-22]-1)*100, '%')
print('63d ret:', (spx.iloc[-1]/spx.iloc[-64]-1)*100, '%')

rv = rt['SPX'].rolling(21).std()*np.sqrt(252)
print('\nSPX 21d realized vol (annualized):', f'{rv.iloc[-1]*100:.1f}%', ' 60d avg:', f'{rv.iloc[-60:].mean()*100:.1f}%')

eq_assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']
eq_rt = rt[eq_assets].mean(axis=1)
print('\nEQ basket 21d mean ret:', f'{eq_rt.iloc[-1]*100:.2f}%')

print('\n=== Macro signals through %s ===' % VIS)
for f in ['DXY','VIX','EURUSD','USDJPY','USDCNY']:
    df = pd.read_csv(f'../persistent/index_data/{f}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= pd.Timestamp(VIS)]
    if len(df)==0:
        print(f, 'no data up to', VIS); continue
    c = df['close']
    r21 = c.iloc[-1]/c.iloc[-22]-1 if len(c)>22 else np.nan
    r63 = c.iloc[-1]/c.iloc[-64]-1 if len(c)>64 else np.nan
    print(f'{f:<8} last={c.iloc[-1]:.3f} 21d={r21*100:7.2f}% 63d={r63*100:7.2f}%')

corr = rt[eq_assets].tail(63).corr()
vals = corr.values[np.triu_indices(len(corr),1)]
print('\nEQ avg pairwise 21d corr (63d):', f'{np.nanmean(vals):.3f}')

# Breadth: % eq assets above 20d MA
above = {}
for a in eq_assets:
    s = px[a].dropna()
    above[a] = s.iloc[-1] > s.rolling(20).mean().iloc[-1]
print('EQ breadth above ma20:', sum(above.values()), '/', len(eq_assets))

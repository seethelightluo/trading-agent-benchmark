import pandas as pd, numpy as np, os, json

SD = '../persistent/stock_data'
ID = '../persistent/index_data'
CUT = pd.Timestamp('2035-03-22')  # visible_through from date.json

watch = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']

def load(p):
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    for c in ['close','open','high','low','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

closes = {}
for s in watch:
    p = os.path.join(SD, s + '.csv')
    if os.path.exists(p):
        closes[s] = load(p)['close']
px = pd.DataFrame(closes).sort_index()
px = px[px.index <= CUT]
print('price panel through', px.index.max().date(), 'shape', px.shape)

ret = px.pct_change()
last = px.index.max()

d = {}
for s in watch:
    c = px[s].dropna()
    if len(c) == 0:
        d[s] = dict(last=np.nan, r20=np.nan, r60=np.nan, r120=np.nan, ma20=np.nan, ma60=np.nan, vol20_ann=np.nan)
        continue
    def retn(n):
        if len(c) > n:
            return c.iloc[-1] / c.iloc[-1-n] - 1
        return np.nan
    r20, r60, r120 = retn(20), retn(60), retn(120)
    ma20 = c.iloc[-1] / c.tail(20).mean() - 1
    ma60 = c.iloc[-1] / c.tail(60).mean() - 1
    v = ret[s].dropna().tail(20).std() * np.sqrt(252)
    d[s] = dict(last=str(c.index.max().date()), r20=round(r20*100,2), r60=round(r60*100,2), r120=round(r120*100,2),
                ma20=round(ma20*100,2), ma60=round(ma60*100,2), vol20_ann=round(v*100,2))

print('\n=== per-asset stats (through %s) ===' % last.date())
for s in watch:
    dd = d[s]
    print(f"{s:10s} last={dd['last']} r20={dd['r20']:7.2f}% r60={dd['r60']:7.2f}% r120={dd['r120']:7.2f}% vsMA20={dd['ma20']:6.2f}% vsMA60={dd['ma60']:6.2f}% vol20={dd['vol20_ann']:5.1f}%")

ab20 = sum(1 for s in watch if d[s]['ma20'] > 0)
ab60 = sum(1 for s in watch if d[s]['ma60'] > 0)
eqw20 = np.nanmean([d[s]['r20'] for s in watch])
eqw60 = np.nanmean([d[s]['r60'] for s in watch])
mean20vol = np.nanmean([d[s]['vol20_ann'] for s in watch])
xr = ret[watch].tail(20)
disp20 = xr.std(axis=1).mean() * 100
disp5 = ret[watch].tail(5).std(axis=1).mean() * 100
disp60 = ret[watch].tail(60).std(axis=1).mean() * 100
print(f'\nBreadth: {ab20}/15 above MA20, {ab60}/15 above MA60')
print(f'20d eqw mean: {eqw20:.3f}% | 60d eqw mean: {eqw60:.3f}%')
print(f'Mean 20d ann vol: {mean20vol:.2f}%')
print(f'Dispersion 20d: {disp20:.3f}% | 5d: {disp5:.3f}% | 60d: {disp60:.3f}%')

for sig in ['VIX','DXY','USDCNY','USDJPY','EURUSD']:
    p = os.path.join(ID, sig + '.csv')
    if os.path.exists(p):
        s = load(p)['close']
        s = s[s.index <= CUT]
        lastv = s.iloc[-1]
        r20 = s.iloc[-1]/s.iloc[-21]-1 if len(s)>21 else np.nan
        r60 = s.iloc[-1]/s.iloc[-61]-1 if len(s)>61 else np.nan
        print(f'{sig}: last={lastv:.2f} ({s.index.max().date()}) 20d={r20*100:+.2f}% 60d={r60*100:+.2f}%')

rank = sorted(watch, key=lambda s: d[s]['r20'] if not np.isnan(d[s]['r20']) else -999)
print('\n20d leaders:', [(s, d[s]['r20']) for s in rank[-4:]])
print('20d laggards:', [(s, d[s]['r20']) for s in rank[:4]])
rank60 = sorted(watch, key=lambda s: d[s]['r60'] if not np.isnan(d[s]['r60']) else -999)
print('60d leaders:', [(s, d[s]['r60']) for s in rank60[-3:]])
print('60d laggards:', [(s, d[s]['r60']) for s in rank60[:3]])

# check flat artifacts at the cut
for s in ['HSI','CN10Y']:
    c = px[s].dropna()
    print(f'{s}: n={len(c)} last={c.iloc[-1]:.4f} last30 unique={len(c.tail(30).unique())} last10 unique={len(c.tail(10).unique())}')

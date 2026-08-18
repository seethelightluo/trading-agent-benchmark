import pandas as pd, numpy as np, glob

ASOF = '2027-06-02'  # last completed trading day before current date 2027-06-03

files = sorted(glob.glob('../persistent/stock_data/*.csv'))
data = {}
for f in files:
    sym = f.split('/')[-1].replace('.csv', '')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    datecol = 'date' if 'date' in df.columns else df.columns[0]
    df[datecol] = pd.to_datetime(df[datecol])
    df = df.sort_values(datecol).set_index(datecol)
    df = df[~df.index.duplicated(keep='last')]
    df = df[df.index <= ASOF]
    data[sym] = df
    print(sym, df.index.min().date(), '->', df.index.max().date(), 'rows:', len(df))

rets = {}
for sym, df in data.items():
    c = df['close']
    last = c.iloc[-1]
    r10 = last / c.iloc[-11] - 1 if len(c) > 11 else np.nan
    r20 = last / c.iloc[-21] - 1 if len(c) > 21 else np.nan
    r60 = last / c.iloc[-61] - 1 if len(c) > 61 else np.nan
    r120 = last / c.iloc[-121] - 1 if len(c) > 121 else np.nan
    vol20 = c.pct_change().iloc[-20:].std() * np.sqrt(252)
    rets[sym] = dict(r10=r10, r20=r20, r60=r60, r120=r120, vol20=vol20)
rdf = pd.DataFrame(rets).T
print('\n', rdf[['r10', 'r20', 'r60', 'r120', 'vol20']].to_string(float_format=lambda x: f'{x:+.4f}'))
print('\nMean r10:', round(rdf.r10.mean(), 4), 'Median r10:', round(rdf.r10.median(), 4))
print('Mean r20:', round(rdf.r20.mean(), 4), 'Median r20:', round(rdf.r20.median(), 4))
print('Mean r60:', round(rdf.r60.mean(), 4), 'Median r60:', round(rdf.r60.median(), 4))
print('Mean vol20:', round(rdf.vol20.mean(), 4), 'Median vol20:', round(rdf.vol20.median(), 4))

# dispersion
px = pd.DataFrame({s: d['close'] for s, d in data.items()})
ret = px.pct_change().dropna()
print('Cross-sectional std of last 20d daily returns:', round(ret.iloc[-20:].std(axis=1).mean(), 4))
print('Avg abs daily cross-sectional mean (last 20d):', round(ret.iloc[-20:].mean(axis=1).abs().mean(), 4))

corr = ret.iloc[-60:].corr()
n = len(corr)
offdiag = corr.values[~np.eye(n, dtype=bool)]
print('Avg pairwise corr (60d):', round(np.nanmean(offdiag), 3))

# trend: MA alignment per asset
print('\n-- trend/MA snapshot --')
for sym in ['SPX', 'NDX', '000300.SH', '000688.SH', 'HSI', 'N225', 'SX5E', 'SOX', 'BTC', 'ETH', 'XAU', 'WTI', 'COPPER', 'US10Y', 'CN10Y']:
    if sym not in data:
        continue
    c = data[sym]['close']
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    slope20 = (ma20 / c.rolling(20).mean().iloc[-21] - 1) if len(c) > 41 else np.nan
    above = 'above' if c.iloc[-1] > ma20 else 'below'
    print(f'{sym:10s} last={c.iloc[-1]:12.2f} ma20={ma20:12.2f} ma60={ma60:12.2f} price_{above}_ma20 slope20={slope20:+.4f}')

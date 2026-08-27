import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
px={s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
p=pd.DataFrame(px).sort_index().loc[:'2035-06-20']; r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252); f=(1/(vol+1e-6)).shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q).dropna(); print(f'H{h}: IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.6f} dates={len(q)} hit={(q>0).mean():.4f}')
print(f'coverage={f.notna().sum(axis=1).mean()/15:.6f} turnover10={(f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift(10)).abs().mean(axis=1).mean():.6f} instruments=15 dates={len(p)}')
fr=p.shift(-10)/p-1
for lo,hi in [(2024,2026),(2027,2029),(2030,2032),(2033,2035)]:
 q=[]
 for dt in f.index:
  if lo<=dt.year<=hi:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q).dropna(); print(f'regime {lo}-{hi}: dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.6f}')

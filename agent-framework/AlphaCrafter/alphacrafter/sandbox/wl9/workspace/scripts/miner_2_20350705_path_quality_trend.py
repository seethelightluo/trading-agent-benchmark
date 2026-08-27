import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
px={s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
p=pd.DataFrame(px).sort_index().loc[:'2035-07-04']; r=p.pct_change()
ret=p.pct_change(60); vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
path=r.rolling(60,min_periods=40).apply(lambda x: abs(np.nansum(x))/(np.nansum(np.abs(x))+1e-12),raw=True)
f=(ret/(vol+1e-8)*path).shift(1)
fr=p.shift(-10)/p-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [5,10,20,40,60]:
 fh=p.shift(-h)/p-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fh.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 s=pd.Series(vals).dropna(); print(f'H{h}: IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1)*np.sqrt(252):.6f} dates={len(s)} hit={(s>0).mean():.4f}')
print(f'coverage={f.notna().sum(axis=1).mean()/15:.6f} turnover10={(f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift(10)).abs().mean(axis=1).mean():.6f} instruments=15 dates={len(p)}')
for lo,hi in [(2020,2023),(2024,2026),(2027,2029),(2030,2032),(2033,2035)]:
 s=q[(q.index.year>=lo)&(q.index.year<=hi)].ic.dropna(); print(f'regime {lo}-{hi}: dates={len(s)} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1)*np.sqrt(252):.6f}')
# signal artifact
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20350705_path_quality_trend_signal.csv',index=False)

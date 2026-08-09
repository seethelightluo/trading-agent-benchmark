import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(D).sort_index().loc[:'2027-03-24']; r=p.pct_change(); mom=p.pct_change(20)
cons=(r.rolling(20).apply(lambda x: np.mean(x>0),raw=True)-.5)*2
fac=mom*cons/(r.rolling(20).std()+1e-8); fwd=p.shift(-1)/p-1
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for a,b in [('all',q),('2020-22',q.loc[:'2022-12-31']),('2023-24',q.loc['2023-01-01':'2024-12-31']),('2025-26',q.loc['2025-01-01':'2026-12-31']),('2027',q.loc['2027-01-01':])]:
 if len(b): print(a,'dates',len(b),'meanIC',b.ic.mean(),'ICIR',b.ic.mean()/b.ic.std(ddof=1),'hit',np.mean(b.ic>0),'avgN',b.n.mean())
print('coverage',fac.notna().sum(axis=1).mean()/15,'turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10]:
 yy=p.shift(-h)/p-1; v=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('horizon',h,'IC',np.nanmean(v),'ICIR',np.nanmean(v)/np.nanstd(v,ddof=1),'dates',len(v))
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20270325_trend_consistency_full_signal.csv',index=False)

import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  x=get_stock_daily_data(s,days=2600)
  if x is not None and len(x):
   x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.sort_values('date').drop_duplicates('date').set_index('date')
 except Exception as e: print('skip',s,str(e))
px=pd.concat({s:d['close'] for s,d in D.items()},axis=1).sort_index(); ret=px.pct_change()
f=(px/px.shift(20)-1)-(px/px.shift(60)-1); f=f/(ret.rolling(20).std()*np.sqrt(20)).replace(0,np.nan)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(px.shift(-1)/px-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
for h in [1,5,10]:
 q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(px.shift(-h)/px-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print('horizon',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('daily dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/len(U),'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
for name,sl in [('2020-22',r.loc['2020':'2022']),('2023-24',r.loc['2023':'2024']),('2025-26',r.loc['2025':'2026'])]: print(name,'dates',len(sl),'IC',sl.ic.mean(),'ICIR',sl.ic.mean()/sl.ic.std(ddof=1),'hit',(sl.ic>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'range',r.index.min(),r.index.max(),'assets',len(D))

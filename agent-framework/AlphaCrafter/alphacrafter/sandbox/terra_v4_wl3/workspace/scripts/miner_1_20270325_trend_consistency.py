import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s, days=2600)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
mom=p.pct_change(20); consistency=(r.rolling(20).apply(lambda x: np.mean(x>0),raw=True)-.5)*2
fac=mom*consistency/(r.rolling(20).std()+1e-8); fwd=p.shift(-1)/p-1
ics=[]; turnovers=[]; prev=None
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
  ranks=fac.loc[dt].rank(pct=True)
  if prev is not None: turnovers.append(abs(ranks-prev).dropna().mean())
  prev=ranks
q=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date')
for a,b in [('all',q),('2020-22',q.loc[:'2022-12-31']),('2023-24',q.loc['2023-01-01':'2024-12-31']),('2025-26',q.loc['2025-01-01':'2026-12-31']),('2027',q.loc['2027-01-01':])]:
 if len(b): print(a,'dates',len(b),'meanIC',b.ic.mean(),'ICIR',b.ic.mean()/(b.ic.std(ddof=1)+1e-12),'hit',np.mean(b.ic>0),'avgN',b.n.mean())
print('coverage',len(q)/len(fac),'turnover',np.mean(turnovers),'range',q.index.min(),q.index.max())
for h in [1,5,10]:
 yy=p.shift(-h)/p-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('horizon',h,'IC',np.nanmean(vals),'n',len(vals))

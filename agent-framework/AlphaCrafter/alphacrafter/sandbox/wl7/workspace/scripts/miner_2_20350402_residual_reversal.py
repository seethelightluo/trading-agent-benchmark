import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
 d=get_stock_daily_data(s,days=4200)
 if d is not None and len(d)>100: xs[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(xs).sort_index().ffill(); r=p.pct_change(); m=r.mean(axis=1)
# residual short-term reversal: negate 5d return after removing each asset's rolling beta to common market
fac=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
for s in p.columns:
 beta=r[s].rolling(60,min_periods=30).cov(m)/m.rolling(60,min_periods=30).var()
 fac[s]=-(p[s].pct_change(5)-beta*m.rolling(5).sum())
fwd=p.shift(-10)/p-1; ics=[]; dates=[]; cov=[]; turns=[]; prev=None
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(q):
   ics.append(q);dates.append(dt);cov.append(len(z)/len(U))
   if prev is not None: turns.append((fac.loc[dt].rank()-prev.rank()).abs().mean()/len(U))
   prev=fac.loc[dt]
ic=pd.Series(ics,index=dates); print('assets',len(xs),'dates',len(ic),'period',ic.index.min(),ic.index.max(),'meanIC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'coverage',np.mean(cov),'turnover',np.mean(turns))
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; q=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(v):q.append(v)
 q=pd.Series(q);print('H',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
for label,(a,b) in {'2020-2024':('2020','2024-12-31'),'2025-2029':('2025','2029-12-31'),'2030-now':('2030','2035-04-02')}.items():
 q=ic.loc[a:b];print(label,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal');out.to_csv('scripts/miner_2_20350402_residual_reversal_signal.csv',index=False)

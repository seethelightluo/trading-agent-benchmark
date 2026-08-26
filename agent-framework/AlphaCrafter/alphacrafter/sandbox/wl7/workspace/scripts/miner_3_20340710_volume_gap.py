import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date')[['open','close','volume']].astype(float)
idx=sorted(set().union(*[set(x.index) for x in D.values()]))
def P(k): return pd.concat({s:x[k] for s,x in D.items()},axis=1).reindex(idx).sort_index()
o,c,v=P('open'),P('close'),P('volume')
r=c.pct_change(); vs=v/v.rolling(20).median()-1
# Fade large overnight/open-to-close shocks, weighted by unusual volume, lagged one day.
f=(-(c/o-1)*(1+vs.clip(lower=-.5,upper=3))).shift(1)
for H in [1,5,10,20]:
 y=c.shift(-H)/c-1; vals=[]; ns=[]
 for dt in c.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(vals).dropna(); print('H%d IC %.8f ICIR %.8f hit %.4f dates %d avgN %.2f'%(H,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
 if H==10:
  for name,n in [('recent180',180),('recent500',500),('recent750',750)]:
   z=q.iloc[-n:]; print(name,'IC %.8f ICIR %.8f hit %.4f dates %d'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),len(z)))
print('period',c.index.min().date(),c.index.max().date(),'rows',len(c),'assets',len(c.columns),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20340710_volume_gap_signal.csv',index=False)
print('artifact rows',len(out))

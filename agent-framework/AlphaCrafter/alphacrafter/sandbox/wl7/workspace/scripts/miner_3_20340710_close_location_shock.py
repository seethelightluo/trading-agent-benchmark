import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date')[['open','high','low','close']].astype(float)
# aligned panels
idx=sorted(set().union(*[set(x.index) for x in P.values()]))
def panel(k): return pd.concat({s:x[k] for s,x in P.items()},axis=1).reindex(idx).sort_index()
o,h,l,c=[panel(k) for k in ['open','high','low','close']]
r=c.pct_change(); rng=(h-l)/o.replace(0,np.nan)
# close-location shock: negative return, close near session low, scaled by relative range; lagged
clv=(2*c-h-l)/(h-l).replace(0,np.nan)
f=(-(r)*((1-clv)/2)*(rng/rng.rolling(20).median()).clip(0,4)).shift(1)
results={}
for H in [1,5,10,20]:
 y=c.shift(-H)/c-1; vals=[]; ns=[]; dates=[]
 for dt in c.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 q=pd.Series(vals,index=dates).dropna(); results[H]=q
 print('H%d IC %.8f ICIR %.8f hit %.4f dates %d avgN %.2f'%(H,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
q=results[10]
for label,n in [('full',len(q)),('recent180',180),('recent500',500),('recent750',750)]:
 z=q if label=='full' else q.iloc[-n:]
 print(label,'H10 IC %.8f ICIR %.8f hit %.4f dates %d'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),len(z)))
rr=f.rank(axis=1,pct=True)
print('period',c.index.min().date(),c.index.max().date(),'rows',len(c),'assets',len(c.columns))
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),rr.diff().abs().mean(axis=1).dropna().mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20340710_close_location_shock_signal.csv',index=False)
print('artifact rows',len(out))

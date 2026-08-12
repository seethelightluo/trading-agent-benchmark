import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,5000)
   if d is not None and len(d)>=100:return d
  except:pass
D={s:get(s) for s in U};D={s:d for s,d in D.items() if d is not None};C=pd.DataFrame({s:d.set_index(pd.to_datetime(d.date)).close.astype(float) for s,d in D.items()}).sort_index();R=C.pct_change()
# Medium-term momentum, penalized by recent volatility acceleration.
m10=C.pct_change(10); v20=R.rolling(20).std(); v60=R.rolling(60).std(); f=m10/(v20*np.sqrt(10)).replace(0,np.nan) * (v60/v20).replace(0,np.nan).clip(0.5,2)
f=f.clip(-5,5);f=f.sub(f.mean(axis=1),axis=0);fr=R.shift(-1);rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('assets',len(D),'dates',len(C),'IC_dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15);print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a,b,len(q),q.mean(),q.mean()/q.std(),(q>0).mean())
for h in [3,5,10]:
 rr=C.pct_change(h).shift(-h);z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,np.nanmean(z),len(z))
q=o.tail(120);print('recent120',q.ic.mean(),q.ic.mean()/q.ic.std(),len(q));f.to_csv('scripts/miner_3_20320722_volscaled_momentum_signal.csv',index_label='date')

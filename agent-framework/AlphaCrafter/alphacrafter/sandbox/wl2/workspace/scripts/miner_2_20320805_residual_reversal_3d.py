import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,5000)
   if d is not None and len(d)>=100:return d
  except:pass
D={s:get(s) for s in U};D={s:d for s,d in D.items() if d is not None};C=pd.DataFrame({s:d.set_index(pd.to_datetime(d.date)).close.astype(float) for s,d in D.items()}).sort_index();R=C.pct_change(); r3=C.pct_change(3);res=r3.sub(r3.median(axis=1),axis=0);v=R.rolling(20).std()*np.sqrt(3);f=(-res/v.replace(0,np.nan)).clip(-3,3);f=f.sub(f.mean(axis=1),axis=0)
def run(h):
 fr=C.pct_change(h).shift(-h);a=[]
 for d in f.index:
  q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:a.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
 return pd.DataFrame(a,columns=['date','ic','n']).set_index('date')
o=run(1);print('assets',len(D),'dates',len(C),'IC_dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15);print('IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(),'hit',(o.ic>0).mean())
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a,b,len(q),q.mean(),q.mean()/q.std())
for h in [3,5,10]:q=run(h);print('decay',h,q.ic.mean(),len(q))
q=o.tail(120).ic;print('recent120',q.mean(),q.mean()/q.std(),len(q));f.to_csv('scripts/miner_2_20320805_residual_reversal_3d_signal.csv',index_label='date')

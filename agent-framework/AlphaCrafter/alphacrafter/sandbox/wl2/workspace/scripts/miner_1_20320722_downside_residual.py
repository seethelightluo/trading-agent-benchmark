import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 try:d=get_index_daily_data(s,5000)
 except:d=None
 if d is None or len(d)<100:
  try:d=get_stock_daily_data(s,5000)
  except:d=None
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d.sort_values('date').drop_duplicates('date').set_index('date')
D={s:get(s) for s in U};D={s:d for s,d in D.items() if d is not None}; C=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index();R=C.pct_change()
r5=C.pct_change(5); med=r5.median(axis=1); res=r5.sub(med,axis=0); vol=R.rolling(30).std()*np.sqrt(5)
# selectively reverse negative residual shocks; positive relative moves receive no alpha
z=(-res/vol.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan); f=z.where(res<0,0.0); f=np.tanh(f); f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('assets',len(D),'dates',len(C),'IC_dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/len(U));print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a,b,len(q),q.mean(),q.mean()/q.std(),(q>0).mean())
for h in [3,5,10]:
 rr=C.pct_change(h).shift(-h);v=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8:v.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,np.nanmean(v),len(v))
q=o.tail(120);print('recent',q.ic.mean(),q.ic.mean()/q.ic.std());f.to_csv('scripts/miner_1_20320722_downside_residual_signal.csv',index_label='date')

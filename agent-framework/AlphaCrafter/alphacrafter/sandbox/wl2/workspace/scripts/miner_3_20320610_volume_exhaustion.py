import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 try:d=get_index_daily_data(s,days=5000)
 except:d=None
 if d is None or len(d)<100:
  try:d=get_stock_daily_data(s,days=5000)
  except:d=None
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d.sort_values('date').drop_duplicates('date').set_index('date')
D={s:get(s) for s in U};D={s:d for s,d in D.items() if d is not None}
C=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index();R=C.pct_change()
V=pd.DataFrame({s:d.volume.astype(float) for s,d in D.items()}).reindex(C.index)
# volume-confirmation exhaustion: negative return with unusually high volume tends to mean short-horizon rebound
vz=np.log(V.replace(0,np.nan)).sub(np.log(V.replace(0,np.nan)).rolling(30).mean()).div(np.log(V.replace(0,np.nan)).rolling(30).std())
f=(-R.rolling(3).sum())*vz.clip(lower=0)
f=f.sub(f.mean(axis=1),axis=0)
out=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:out.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'price_dates',len(C),'IC_dates',len(o),'avg_n',round(o.n.mean(),3),'coverage',round(o.n.mean()/len(U),4));print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a+'-'+b,len(q),q.mean(),q.mean()/q.std())
for h in [3,5,10]:
 rr=C.pct_change(h).shift(-h);v=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:v.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,np.nanmean(v),len(v))
print('recent',o.tail(120).ic.mean(),o.tail(120).ic.mean()/o.tail(120).ic.std())
f.to_csv('scripts/miner_3_20320610_volume_exhaustion_signal.csv',index_label='date')

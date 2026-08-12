import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 try:d=get_index_daily_data(s,days=5000)
 except:d=None
 if d is None or len(d)<100:
  try:d=get_stock_daily_data(s,days=5000)
  except:d=None
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d.sort_values('date').drop_duplicates('date').set_index('date')
D={s:fetch(s) for s in U};D={s:d for s,d in D.items() if d is not None}
C=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); R=C.pct_change()
# Cross-asset residual mean reversion: each asset's 20d return relative to
# contemporaneous cross-sectional median, scaled by its 60d volatility.
ret20=C.pct_change(20); med=ret20.median(axis=1); resid=ret20.sub(med,axis=0)
vol60=R.rolling(60).std()*np.sqrt(20)
f=(-resid/vol60.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
# damp extreme observations to avoid a single crypto shock dominating ranks
f=np.sign(f)*np.log1p(np.abs(f)); f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'price_dates',len(C),'IC_dates',len(o),'avg_n',round(o.n.mean(),3),'coverage',round(o.n.mean()/len(U),4))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic; print(a+'-'+b,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
for h in [1,3,5,10]:
 rr=C.pct_change(h).shift(-h);v=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:v.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,'IC %.6f n %d'%(np.nanmean(v),len(v)))
q=o.tail(120);print('recent120 IC %.6f ICIR %.6f n %d'%(q.ic.mean(),q.ic.mean()/q.ic.std(),len(q)))
f.to_csv('scripts/miner_3_20320708_residual_mr_signal.csv',index_label='date')

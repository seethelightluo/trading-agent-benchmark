import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=None
 try:d=get_index_daily_data(s,days=5000)
 except Exception: pass
 if d is None or len(d)<100:
  try:d=get_stock_daily_data(s,days=5000)
  except Exception: pass
 if d is None:return None
 return d.assign(date=pd.to_datetime(d.date)).sort_values('date').drop_duplicates('date').set_index('date')
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
C=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); R=C.pct_change()
# Lagged relative-strength acceleration: recent 10d trend versus prior 20d trend,
# scaled by trailing volatility, with cross-sectional demeaning.
r10=C.pct_change(10); prior20=C.pct_change(30)-r10
vol=R.rolling(30).std()*np.sqrt(10)
f=((r10-prior20)/vol.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: rows.append((d,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'price_dates',len(C),'IC_dates',len(o),'avg_n',round(o.n.mean(),3),'coverage',round(o.n.mean()/len(U),4))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic; print(a+'-'+b,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
for h in [3,5,10]:
 rr=C.pct_change(h).shift(-h);v=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:v.append(f.loc[d].corr(rr.loc[d],method='spearman'))
 print('decay',h,'IC %.6f n %d'%(np.nanmean(v),len(v)))
q=o.tail(120);print('recent120 IC %.6f ICIR %.6f n %d'%(q.ic.mean(),q.ic.mean()/q.ic.std(),len(q)))
f.to_csv('scripts/miner_1_20320805_rs_accel_signal.csv',index_label='date')

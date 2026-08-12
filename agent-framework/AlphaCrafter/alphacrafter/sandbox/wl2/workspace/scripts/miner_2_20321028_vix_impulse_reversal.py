import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,5000)
   if x is not None and len(x)>=100:return x
  except Exception: pass
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items()}).sort_index(); R=C.pct_change()
v=get_index_daily_data('VIX',5000); V=v.set_index(pd.to_datetime(v.date)).close.astype(float).reindex(C.index).ffill()
# Lagged cross-asset residual reversal activated for five sessions after an unusually sharp 3-day VIX impulse.
res=R.sub(R.mean(axis=1),axis=0); shock=res.rolling(5,min_periods=4).sum().shift(1)
vret=V.pct_change(3).shift(1); threshold=vret.rolling(252,min_periods=100).quantile(.70).shift(1)
active=(vret>threshold).astype(float).rolling(5,min_periods=1).max()
f=(-shock/R.rolling(20,min_periods=10).std().shift(1).replace(0,np.nan)).mul(active.replace(0,np.nan),axis=0)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=o.ic
print('assets',len(D),'dates',len(C),'IC_dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean()))
for a,b in [('2026','2029'),('2030','2032')]:
 q=ic.loc[a:b]; print(a,b,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
for h in [3,5,10]:
 rr=C.pct_change(h).shift(-h); z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,np.nanmean(z),len(z))
q=ic.tail(120); print('recent120 %.6f %.6f'%(q.mean(),q.mean()/q.std()))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_2_20321028_vix_impulse_reversal_signal.csv',index_label='date')

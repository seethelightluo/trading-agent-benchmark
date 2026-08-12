import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   z=fn(s,5000)
   if z is not None and len(z)>=100:return z
  except Exception: pass
D={s:load(s) for s in U};D={s:z for s,z in D.items() if z is not None}
C=pd.DataFrame({s:z.set_index(pd.to_datetime(z.date)).close.astype(float) for s,z in D.items()}).sort_index(); C=C.groupby(level=0).last(); R=C.pct_change()
v=get_index_daily_data('VIX',5000); V=v.set_index(pd.to_datetime(v.date)).close.astype(float).reindex(C.index).ffill()
# Candidate: reversal after synchronized downside capitulation, gated by lagged VIX impulse.
res=R.sub(R.mean(axis=1),axis=0)
shock=res.rolling(3,min_periods=3).sum().shift(1)
breadth=(R<0).mean(axis=1).rolling(3,min_periods=3).mean().shift(1)
vimp=V.pct_change(3).shift(1); vcut=vimp.rolling(252,min_periods=100).quantile(.70).shift(1)
active=((breadth>.60)&(vimp>vcut)).astype(float)
vol=R.rolling(20,min_periods=10).std().shift(1)
base=(-shock/vol.replace(0,np.nan)); f=base.where(active.astype(bool),np.nan); print('active_rows',int(active.sum()),'finite',int(np.isfinite(f.to_numpy()).sum()))
rows=[]
for i,d in enumerate(f.index[:-1]):
 q=pd.concat([f.iloc[i],R.iloc[i+1]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=o.ic
print('assets',len(D),'dates',len(C),'IC_dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean()))
for a,b in [('2020','2025'),('2026','2029'),('2030','2032')]:
 q=ic.loc[a:b];print(a,b,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
for h in [1,3,5,10]:
 rr=R.shift(-h).rolling(h).sum() if h>1 else R.shift(-1)
 vals=[]
 for i,d in enumerate(f.index[:-h]):
  q=pd.concat([f.iloc[i],rr.iloc[i+h]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,len(vals),np.nanmean(vals))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_3_20321028_vix_breadth_capitulation_signal.csv',index_label='date')

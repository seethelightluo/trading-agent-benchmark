import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,5000)
   if x is not None and len(x)>100:return x
  except: pass
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items()}).sort_index().groupby(level=0).last()
R=C.pct_change(); vol=R.rolling(20,min_periods=15).std().shift(1)
# Novel candidate: acceleration of medium-term trend, scaled by recent risk; all inputs lagged.
mom20=C.pct_change(20).shift(1); mom60=C.pct_change(60).shift(1)
# reward trend improvement while avoiding assets whose recent trend is already reversing
f=(mom20-mom60/3)/vol.replace(0,np.nan)
# market regime gate: only use acceleration when cross-asset median 20d trend is nonnegative
reg=(mom20.median(axis=1)>=0)
f=f.where(reg, -f) # defensive reversal in bearish regime, remains interpretable
rows=[]
for i,d in enumerate(f.index[:-1]):
 q=pd.concat([f.iloc[i],R.iloc[i+1]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=o.ic
print('assets',len(D),'dates',len(C),'IC_dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean()))
for a,b in [('2020','2025'),('2026','2029'),('2030','2032')]:
 q=ic.loc[a:b];print(a,b,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
for h in [3,5,10]:
 rr=R.shift(-h).rolling(h).sum(); vals=[]
 for i in range(len(f)-h):
  q=pd.concat([f.iloc[i],rr.iloc[i+h]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,len(vals),np.nanmean(vals))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_2_20321111_accel_regime_signal.csv',index_label='date')

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
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items()}).sort_index(); R=C.pct_change(); res=R.sub(R.mean(axis=1),axis=0)
# Lagged 1-day residual reversal, volatility normalized with a 20-day trailing estimate.
f=-res.shift(1)/R.rolling(20).std().shift(1).replace(0,np.nan)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'price_dates',len(C),'IC_dates',len(o),'avg_n %.3f coverage %.4f'%(o.n.mean(),o.n.mean()/15))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic; print(a,b,len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
for h in [3,5,10]:
 rr=C.pct_change(h).shift(-h); z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,'IC %.6f n %d'%(np.nanmean(z),len(z)))
q=o.tail(120); print('recent120 IC %.6f ICIR %.6f n %d'%(q.ic.mean(),q.ic.mean()/q.ic.std(),len(q)))
print('turnover_proxy %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_3_20320916_residual1_signal.csv',index_label='date')

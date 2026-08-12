import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   z=fn(s,5000)
   if z is not None and len(z)>100:return z
  except: pass
D={s:load(s) for s in U}; D={s:z for s,z in D.items() if z is not None}
C=pd.DataFrame({s:z.set_index(pd.to_datetime(z.date)).close.astype(float) for s,z in D.items()}).sort_index().groupby(level=0).last(); R=C.ffill().pct_change()
v=get_index_daily_data('VIX',5000); V=v.set_index(pd.to_datetime(v.date)).close.astype(float).reindex(C.index).ffill()
res=R.sub(R.mean(axis=1),axis=0); shock=res.rolling(3,min_periods=3).sum().shift(1)
breadth=(R<0).mean(axis=1).rolling(3,min_periods=3).mean().shift(1); vi=V.pct_change(3).shift(1); cut=vi.rolling(252,min_periods=100).quantile(.70).shift(1)
active=(breadth>.60)&(vi>cut); vol=R.rolling(20,min_periods=10).std().shift(1); f=(-shock/vol).where(active,np.nan)
for h in [1,3,5,10]:
 fr=R.rolling(h).sum().shift(-h)
 rows=[]
 for d in f.index:
  q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 a=pd.Series(rows).dropna(); print('H',h,'dates',len(a),'mean',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
print('assets',len(D),'calendar_dates',len(C),'active',int(active.sum()),'coverage_active',f.notna().sum(axis=1).replace(0,np.nan).mean()/len(D))
# signal artifact
f.to_csv('scripts/miner_3_20321111_vix_breadth_capitulation_signal.csv',index_label='date')

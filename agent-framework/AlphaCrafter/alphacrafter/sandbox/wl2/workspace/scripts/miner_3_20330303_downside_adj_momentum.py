import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   z=fn(s,5000)
   if z is not None and len(z)>100:return z
  except Exception: pass
D={s:load(s) for s in U}; D={s:z for s,z in D.items() if z is not None}
px=pd.DataFrame({s:z.set_index(pd.to_datetime(z.date)).close.astype(float) for s,z in D.items()}).sort_index().groupby(level=0).last()
r=px.pct_change(); down=r.where(r<0).rolling(60,min_periods=30).std()*np.sqrt(252)
sig=((px/px.shift(20)-1)/down).shift(1)
for h in [1,3,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals).dropna(); print('H',h,'N',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
cov=sig.notna().sum(axis=1)/len(U); rank=sig.rank(axis=1,pct=True); turn=(rank-rank.shift()).abs().mean(axis=1).dropna().mean()
print('dates',len(px),'assets',len(D),'coverage',cov.mean(),'valid>=8',sum(sig.notna().sum(axis=1)>=8),'turnover',turn)
sig.to_csv('scripts/miner_3_20330303_downside_adj_momentum_signal.csv')

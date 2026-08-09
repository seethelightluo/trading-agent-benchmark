import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for a in A:
 d=get_stock_daily_data(a,days=1800)
 if d is not None and len(d)>100: px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); x=p.pct_change(3); breadth=pd.DataFrame({a:(x.gt(0).sum(axis=1)-x[a].gt(0).astype(int))/14 for a in px})
peer=p.pct_change(5).copy(); peer=pd.DataFrame({a:peer.drop(columns=a).median(axis=1) for a in px})
rev=-p.pct_change(5); mom=p.pct_change(20)/p.pct_change(20).rolling(20).std()
for name,z in [('peer',peer),('reversal',rev),('momentum',mom)]:
 q=pd.concat([breadth.stack().rename('b'),z.stack().rename(name)],axis=1).dropna(); print(name,round(q.b.corr(q[name]),4),len(q))

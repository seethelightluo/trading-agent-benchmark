import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in U:
 d=get_stock_daily_data(a,days=3000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d.date); d=d.set_index('date'); D[a]=d.close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# lagged multi-horizon volatility-adjusted momentum rank ensemble
v20=r.rolling(20).std().shift(1); v60=r.rolling(60).std().shift(1)
m5=p.pct_change(5).shift(1)/v20
m20=p.pct_change(20).shift(1)/v20
m60=p.pct_change(60).shift(1)/v60
f=(m5.rank(axis=1,pct=True)+m20.rank(axis=1,pct=True)+m60.rank(axis=1,pct=True))/3
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 x=pd.Series(vals).dropna(); print('h',h,'dates',len(x),'avg_n',round(len(q),2),'IC %.6f ICIR %.6f hit %.3f'%(x.mean(),x.mean()/x.std(),(x>0).mean()))
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'cutoff',p.index.max(),'assets',len(D))

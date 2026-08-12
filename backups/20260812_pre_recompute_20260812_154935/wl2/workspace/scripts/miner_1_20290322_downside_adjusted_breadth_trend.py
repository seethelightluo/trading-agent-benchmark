import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); F[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(F).sort_index().ffill(limit=3); r=np.log(px).diff()
# Downside-risk-adjusted trend: medium trend divided by downside deviation,
# with a cross-sectional breadth gate that suppresses pro-risk signals in broad stress.
ret=r.rolling(30).sum().shift(1)
down=r.where(r<0,0).rolling(30).std().shift(1)
base=ret/(down+1e-6)
breadth=(r.rolling(20).sum().shift(1)>0).mean(axis=1)
# In broad stress, favor the least negative risk-adjusted trends; otherwise normal trend.
f=base.mul(np.where(breadth<0.35,-0.5,1.0),axis=0)
for h in [1,5,10,20]:
 fr=np.log(px).shift(-h)-np.log(px); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 x=pd.Series(vals).dropna(); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round(np.mean(x>0),4))
q=f.rank(axis=1,pct=True); print('turnover',round(q.diff().abs().mean(axis=1).dropna().mean(),6),'assets',len(px.columns),'dates',len(px),'start',px.index.min(),'end',px.index.max())
f.to_csv('scripts/miner_1_20290322_downside_adjusted_breadth_trend_signal.csv')

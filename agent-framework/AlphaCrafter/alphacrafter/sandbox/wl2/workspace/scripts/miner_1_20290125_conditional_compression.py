import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);F[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(F).sort_index().ffill(limit=3); r=np.log(px).diff()
v10=r.rolling(10).std();v60=r.rolling(60).std(); comp=-(v10/v60).shift(1)
# Conditional compression: retain compression ranking only when prior 20d return is positive;
# otherwise use a mild defensive inverse-volatility score. All inputs lagged.
trend=np.log(px).diff(20).shift(1); invvol=-(v60.shift(1))
f=comp.where(trend>0,invvol)
for h in [5,10,20]:
 fr=np.log(px).shift(-h)-np.log(px); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=pd.Series(vals).dropna();print('H',h,'dates',len(x),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',np.mean(x>0))
rank=f.rank(axis=1,pct=True);print('turnover_proxy',rank.diff().abs().mean(axis=1).dropna().mean(),'assets',len(px.columns),'dates',len(px),'start',px.index.min(),'end',px.index.max())
f.to_csv('scripts/miner_1_20290125_conditional_compression_signal.csv')

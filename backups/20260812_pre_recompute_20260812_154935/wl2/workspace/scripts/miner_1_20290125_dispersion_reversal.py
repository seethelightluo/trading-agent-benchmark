import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];F={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=4000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);F[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(F).sort_index().ffill(limit=3);r=np.log(px).diff()
# Dispersion-conditioned reversal: short-term reversal is expected to be strongest
# after unusually dispersed cross-asset moves; otherwise use medium trend.
disp=r.rolling(5).std().mean(axis=1).shift(1); threshold=disp.rolling(120).quantile(.7).shift(1)
rev=-r.rolling(3).sum().shift(1); mom=r.rolling(20).sum().shift(1)
f=rev.where(disp>threshold,mom)
for h in [5,10,20]:
 fr=np.log(px).shift(-h)-np.log(px);vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=pd.Series(vals).dropna();print('H',h,'dates',len(x),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252),'hit',np.mean(x>0))
print('turnover_proxy',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'assets',len(px.columns),'dates',len(px),'start',px.index.min(),'end',px.index.max())
f.to_csv('scripts/miner_1_20290125_dispersion_reversal_signal.csv')

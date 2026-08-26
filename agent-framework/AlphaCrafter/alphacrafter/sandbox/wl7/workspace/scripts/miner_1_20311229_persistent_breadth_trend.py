import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=px.pct_change()
# Persistent breadth-conditioned medium trend. Require breadth state to persist via 5-day mean,
# then apply lagged 40d trend with regime direction; all observations lagged.
breadth=(r.rolling(20,min_periods=15).mean()>0).mean(axis=1)
reg=np.where(breadth.rolling(5,min_periods=5).mean()>=0.5,1,-1)
reg=pd.Series(reg,index=px.index).shift(1)
f=px.pct_change(40).mul(reg,axis=0).shift(1)
print('span',px.index.min(),px.index.max(),'assets',len(px.columns))
for h in [1,5,10,20]:
 fr=px.pct_change(h).shift(-h); a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):a.append(q);ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'cov',round(np.mean(ns)/15,4))
for cut,name in [(0.33,'early'),(0.66,'mid')]: pass
rank=f.rank(axis=1,pct=True)
print('turnover_proxy',round(float(rank.diff().abs().mean(axis=1).dropna().mean()),6))
f.to_csv('scripts/miner_1_20311229_persistent_breadth_trend_signal.csv')

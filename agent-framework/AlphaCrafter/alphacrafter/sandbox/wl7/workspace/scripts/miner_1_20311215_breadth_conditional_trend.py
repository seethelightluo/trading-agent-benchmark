import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=px.pct_change()
# Conditional trend: own 30d momentum, activated when broad cross-asset breadth is positive;
# when breadth is negative, reverse the signal. All inputs lagged one completed day.
breadth=(r.rolling(20,min_periods=15).mean()>0).mean(axis=1)
reg=(2*(breadth>=0.5)-1).shift(1)
f=(px.pct_change(30).mul(reg,axis=0)).shift(1)
print('span',px.index.min(),px.index.max(),'assets',len(px.columns))
for h in [1,5,10,20]:
 fr=px.pct_change(h).shift(-h); a=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):a.append(q);ns.append(len(z));ds.append(dt)
 a=np.array(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'cov',round(np.mean(ns)/15,4))
# proxy turnover and regime splits
rank=f.rank(axis=1,pct=True); print('turnover_proxy',round(float(rank.diff().abs().mean(axis=1).dropna().mean()),6))
# signal artifact for provenance
f.to_csv('scripts/miner_1_20311215_breadth_conditional_trend_signal.csv')

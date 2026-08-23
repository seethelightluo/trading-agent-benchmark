import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,3000) for s in U}
px=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index().ffill(); px=px.loc[:pd.Timestamp('2030-06-26')]
r=px.pct_change(); dates=px.index
mom=px.pct_change(20); vol=r.rolling(20).std()*np.sqrt(252); dd=px/px.rolling(20).max()-1; f=mom/(vol+1e-8)+0.5*dd
for h in [5,10,20]:
 a=[]; ns=[]; turns=[]
 for i in range(20,len(dates)-h):
  z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
   if i>20: turns.append((f.iloc[i].rank(pct=True)-f.iloc[i-1].rank(pct=True)).abs().mean())
 a=np.array(a); print('horizon',h,'dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,5),'IC',round(a.mean(),8),'ICIR',round(a.mean()/(a.std(ddof=1)/np.sqrt(len(a))),5),'hit',round(np.mean(a>0),5),'turnover',round(np.mean(turns),6))
ser=[]
for i in range(20,len(dates)-10):
 z=pd.concat([f.iloc[i],px.iloc[i+10]/px.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:ser.append((dates[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
for yr in sorted(set(x[0].year for x in ser)):
 q=[v for d,v in ser if d.year==yr];print('regime',yr,len(q),round(np.mean(q),6))
print('date_range',dates[0],dates[-1],'rows',len(dates))

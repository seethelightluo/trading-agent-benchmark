import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=np.log(px).diff()
ret20=px.pct_change(20); market=ret20.median(axis=1); vol40=r.rolling(40).std()*np.sqrt(252)
f=-(ret20.sub(market,axis=0)).div(vol40).replace([np.inf,-np.inf],np.nan).shift(1)
rows=[]
for h in [1,5,10,20]:
 fr=px.pct_change(h).shift(-h); ics=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(ic): ics.append(ic);ns.append(len(z));dates.append(dt)
 a=np.array(ics); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4))
 if h==10: pd.DataFrame({'date':dates,'ic':a,'n':ns}).to_csv('scripts/miner_1_20311117_residual_reversal_ic.csv',index=False)
f.to_csv('scripts/miner_1_20311117_residual_reversal_signal.csv'); print('span',px.index.min(),px.index.max(),'assets',len(px.columns))

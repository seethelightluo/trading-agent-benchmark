import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}; p=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}); v=pd.DataFrame({s:d.set_index('date')['volume'] for s,d in D.items() if d is not None})
r=p.pct_change(); vol=r.rolling(20).std()*np.sqrt(252); shock=(v.rolling(5).mean()/v.rolling(60).mean()).clip(.5,2)
# reversal is stronger when unusual participation confirms an oversold move
sig=(-r.rolling(5).sum()/(vol+1e-8))*shock; sig=sig.shift(1)
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in sig.index:
  x=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(x)>=8: vals.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'));ns.append(len(x))
 z=pd.Series(vals);print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('coverage',sig.notna().sum().sum()/sig.size); rk=sig.rank(axis=1,pct=True);print('turnover10',np.nanmean([(rk.iloc[i]-rk.iloc[i-10]).abs().mean() for i in range(10,len(rk))]))
f=p.shift(-10)/p-1;o=[]
for dt in sig.index:
 x=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(x)>=8:o.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'))
q=len(o)//3;print('thirds',[round(np.mean(o[j*q:(j+1)*q]),6) for j in range(3)])
sig.stack().rename('signal').to_csv('scripts/miner_2_20320823_volume_shock_reversal_signal.csv',header=True)

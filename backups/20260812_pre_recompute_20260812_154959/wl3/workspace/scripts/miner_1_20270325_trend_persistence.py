import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs=[]
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150:d=get_index_daily_data(s,2600)
 if d is not None: xs.append(d[['date','close']].assign(symbol=s))
w=pd.concat(xs).pivot(index='date',columns='symbol',values='close').sort_index(); r=w.pct_change()
# Trend persistence: medium return, conditioned on fraction of positive sessions, volatility normalized.
vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
persist=(r>0).rolling(40,min_periods=25).mean()
f=w.pct_change(40)/(vol+1e-12)*(0.5+persist)
lo=f.quantile(.05,axis=1); hi=f.quantile(.95,axis=1)
f=f.clip(lower=lo,upper=hi,axis=0)
print('cutoff',w.index.max().date(),'dates',len(w),'assets',len(w.columns))
for h in [1,3,5,10]:
 z=[]; ns=[]; fr=w.shift(-h)/w-1
 for dt in w.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=pd.Series(z); print('H',h,'n',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1)*np.sqrt(len(z)),6),'hit',round((z>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
f.stack().rename('signal').reset_index().to_csv('scripts/miner_1_20270325_trend_persistence_signal.csv',index=False)

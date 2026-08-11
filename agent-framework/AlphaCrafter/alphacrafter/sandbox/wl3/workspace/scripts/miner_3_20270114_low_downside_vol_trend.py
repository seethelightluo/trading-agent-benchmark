import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
a=[]
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150:d=get_index_daily_data(s,2600)
 if d is not None:a.append(d[['date','close']].assign(symbol=s))
w=pd.concat(a).pivot(index='date',columns='symbol',values='close').sort_index(); r=w.pct_change()
# Volatility-regime persistence: favor assets with low recent downside volatility,
# but only when their 20d trend is nonnegative; otherwise use the defensive low-vol score.
v=r.rolling(20,min_periods=15).std(); down=r.where(r<0).rolling(20,min_periods=15).std()
trend=w.pct_change(20); f=(-down/(v+1e-12)) + 0.25*trend/(v*np.sqrt(20)+1e-12)
# rank IC, next-day return
for h in [1,3,5,10]:
 z=[]; ns=[]
 for dt in w.index:
  q=pd.concat([f.loc[dt],(w.shift(-h)/w-1).loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 q=pd.Series(z).dropna();print('H',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean(),'avgN',np.mean(ns))
print('coverage',f.notna().mean().mean(),'turnover',((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift()).abs().mean(axis=1)).mean())
out=f.stack().rename('signal').reset_index();out.to_csv('scripts/miner_3_20270114_low_downside_vol_trend_signal.csv',index=False)
print('cutoff',w.index.max().date(),'dates',len(w),'assets',len(w.columns))

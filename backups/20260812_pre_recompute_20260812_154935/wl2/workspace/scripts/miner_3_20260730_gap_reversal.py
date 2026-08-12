import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=pd.Index(sorted(set.intersection(*[set(x.index) for x in D.values()])))
P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); O=pd.DataFrame({s:D[s].open.reindex(dates) for s in U})
# Overnight gap mean reversion: negative recent open-to-prior-close gap predicts next return; 3d smoothing lowers noise.
gap=O/P.shift(1)-1
F=(-gap.rolling(3,min_periods=3).mean()).shift(1)
print('gap_reversal_3d universe',len(U),'dates',len(dates))
for h in [1,5,10]:
 Y=P.shift(-h).div(P).sub(1); q=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.asarray(q);print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for k in [252,504]:
   x=q[-k:];print('recent',k,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))

import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}; dates=D['SPX'].index
P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change(); breadth=R.gt(0).rolling(12,min_periods=9).mean()-R.lt(0).rolling(12,min_periods=9).mean(); market=R.median(axis=1).rolling(5,min_periods=4).sum(); F=(breadth* np.sign(market).replace(0,np.nan).ffill()).shift(1)
print('idea regime_signed_trend_consistency; universe',len(U),'dates',len(dates))
for h in [1,5,10]:
 Y=P.shift(-h).div(P).sub(1);q=[];ns=[];ds=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.asarray(q); print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   x=q[[d.year==yr for d in ds]]; print('regime',yr,len(x),round(x.mean(),6) if len(x) else None,round(x.mean()/x.std(ddof=1),4) if len(x)>1 else None)
  for k in [252,504]:
   x=q[-k:]; print('recent',k,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))

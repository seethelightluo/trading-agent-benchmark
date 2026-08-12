import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}; dates=D['SPX'].index
R=pd.DataFrame({s:D[s].close.pct_change().reindex(dates) for s in U}); Y=pd.DataFrame({s:D[s].close.shift(-1).div(D[s].close).sub(1).reindex(dates) for s in U})
for n in [6,9,15,20,30]:
 # persistence score: fraction positive minus fraction negative, lagged
 F=(R.gt(0).rolling(n,min_periods=max(4,n//2)).mean()-R.lt(0).rolling(n,min_periods=max(4,n//2)).mean()).shift(1)
 q=[];ns=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.array(q);print('window',n,'dates',len(q),'meanN',round(np.mean(ns),2),'coverage',round(F.notna().sum().sum()/F.size,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'recent252',round(q[-252:].mean(),6),round(q[-252:].mean()/q[-252:].std(ddof=1),6))

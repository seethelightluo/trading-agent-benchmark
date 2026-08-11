import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
clv={s:-(2*(D[s].close-D[s].low)/(D[s].high-D[s].low).replace(0,np.nan)) for s in U}; mom={s:D[s].close.pct_change(20) for s in U}
for w in [0.10,0.15,0.20,0.25,0.50]:
 F=pd.DataFrame({s:(clv[s].rank(pct=True)+w*mom[s].rank(pct=True)).shift(1) for s in U})
 Y=pd.DataFrame({s:D[s].close.shift(-1)/D[s].close-1 for s in U});q=[];ns=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.asarray(q); print('w',w,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4))

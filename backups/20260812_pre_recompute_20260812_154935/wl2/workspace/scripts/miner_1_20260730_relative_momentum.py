import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
# Calculate on each instrument's own completed-session sequence, then lag one observation.
F=pd.DataFrame({s:(D[s].close.pct_change().rolling(20).sum().sub(D[s].close.pct_change().rolling(20).sum().median()) if False else D[s].close.pct_change().rolling(20).sum()).shift(1) for s in U})
# cross-sectional demeaning is applied on each date after lagging, avoiding sparse calendar rolling artifacts
F=F.sub(F.median(axis=1),axis=0)
for h in [1,5,10]:
 Y=pd.DataFrame({s:D[s].close.shift(-h)/D[s].close-1 for s in U}); q=[];ns=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.array(q); print('h',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))

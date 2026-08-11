import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
# Smoothed overnight-gap reversal: negative rolling mean of completed open/prior-close gaps.
gap=pd.DataFrame({s:D[s].open/D[s].close.shift(1)-1 for s in U}).sort_index()
F=-gap.rolling(3,min_periods=3).mean()
Y={h:pd.DataFrame({s:D[s].close.shift(-h)/D[s].close-1 for s in U}).sort_index() for h in [1,5,10]}
for h in [1,5,10]:
 q=[];ns=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y[h].loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.asarray(q); print('horizon',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for k in [252,504,756]:
   x=q[-k:];print('recent',k,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for yr in range(2020,2027):
 q=[]
 for dt in F.loc[str(yr)].index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y[1].loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic)
 print('regime',yr,'dates',len(q),'IC',round(np.mean(q),6) if q else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))

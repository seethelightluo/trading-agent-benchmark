import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=sorted(set.intersection(*[set(D[s].index) for s in U])); C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=C.pct_change()
# Trend breadth: fraction of positive sessions over 20, centered and lagged.
F=R.gt(0).rolling(20,min_periods=15).mean().sub(.5).shift(1)
for h in [1,5,10]:
 Y=C.shift(-h).div(C).sub(1); q=[]; ns=[]
 for dt in dates:
  z=pd.concat([F.loc[[dt]].T,Y.loc[[dt]].T],axis=1).dropna(); z.columns=['f','y']
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.array(q);print('horizon',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   a=[]
   for dt in [d for d in dates if d.year==yr]:
    z=pd.concat([F.loc[[dt]].T,Y.loc[[dt]].T],axis=1).dropna(); z.columns=['f','y']
    if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic)
   print('regime',yr,'dates',len(a),'IC',round(np.mean(a),6) if a else None,'ICIR',round(np.mean(a)/np.std(a,ddof=1),4) if len(a)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))

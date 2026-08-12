import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index
P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change()
# Cross-asset residual momentum: each asset's 20d return minus the contemporaneous
# cross-sectional median 20d return, lagged one day to avoid look-ahead.
raw=P.pct_change(20); F=raw.sub(raw.median(axis=1),axis=0).shift(1)
for h in [1,5,10]:
 Y=P.shift(-h).div(P).sub(1); q=[]; ns=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:
   v=spearmanr(z.f,z.y).statistic
   if np.isfinite(v): q.append(v); ns.append(len(z))
 q=np.array(q)
 print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   x=[]
   for dt in dates[dates.year==yr]:
    z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
    if len(z)>=8:
     v=spearmanr(z.f,z.y).statistic
     if np.isfinite(v):x.append(v)
   print('regime',yr,'dates',len(x),'IC',round(np.mean(x),6) if x else None,'ICIR',round(np.mean(x)/np.std(x,ddof=1),4) if len(x)>1 else None)
rank=F.rank(axis=1,pct=True); print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4))

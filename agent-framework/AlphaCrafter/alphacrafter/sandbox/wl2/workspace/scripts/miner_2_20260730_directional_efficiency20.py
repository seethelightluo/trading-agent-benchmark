import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
dates=pd.Index(sorted(set.intersection(*[set(x.index) for x in D.values()])))
P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change()
# Lagged 20-session directional efficiency: signed displacement divided by path length.
F=(P.pct_change(20)/(R.abs().rolling(20,min_periods=15).sum())).shift(1)
print('efficiency20 universe',len(U),'dates',len(dates))
for h in [1,5,10]:
 Y=P.shift(-h).div(P).sub(1); q=[]; ns=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.asarray(q); print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for k in [252,504]:
   x=q[-k:]; print('recent',k,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for yr in sorted(set(dates.year)):
 vals=[]
 for dt in dates[dates.year==yr]:
  z=pd.DataFrame({'f':F.loc[dt],'y':P.pct_change().shift(-1).loc[dt]}).dropna()
  if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic)
 print('regime',yr,'dates',len(vals),'IC',round(np.mean(vals),6) if vals else None)

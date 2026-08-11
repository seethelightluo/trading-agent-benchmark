import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in D.values()]))
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U});
# Momentum acceleration: short (5d) return relative to medium (20d) return, lagged one completed day.
R5=C/C.shift(5)-1; R20=C/C.shift(20)-1
F=(R5-R20/4).shift(1)
for h in [1,5,10]:
 Y=C.shift(-h)/C-1; vals=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 a=np.array(vals); print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==1:
  for n in [252,504]:
   q=a[-n:];print('recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'date_range',dates[0],dates[-1])
for yr in range(2020,2027):
 q=np.array([v for d,v in zip(ds,vals) if d.year==yr]);
 if len(q): print('regime',yr,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
out=F.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20260730_momentum_acceleration_signal.csv',index=False)

import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in D.values()]))
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=C.pct_change()
# Directional efficiency: cumulative 20-session return / sum absolute daily returns, lagged one session.
F=(C/C.shift(20)-1)/(R.abs().rolling(20,min_periods=18).sum()+1e-12); F=F.shift(1)
for h in [1,5,10]:
 Y=C.shift(-h)/C-1; vals=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 a=np.array(vals); print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==1:
  print('recent252',round(a[-252:].mean(),6),round(a[-252:].mean()/a[-252:].std(ddof=1),6),'recent504',round(a[-504:].mean(),6),round(a[-504:].mean()/a[-504:].std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'date_range',dates[0],dates[-1])
for yr in [2020,2021,2022,2023,2024,2025,2026]:
 z=[v for v,d in zip(vals,ds) if d.year==yr] if h==10 else []
 # no-op; daily regime separately below
A=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:A.append((dt,spearmanr(z.f,z.y).statistic))
for yr in range(2020,2027):
 q=np.array([v for d,v in A if d.year==yr]);
 if len(q): print('regime',yr,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
# signal artifact for audit
out=F.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20260730_trend_efficiency_signal.csv',index=False)

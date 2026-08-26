import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2031-06-11')
d={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date'); d[s]=x.loc[:END,'close']
p=pd.DataFrame(d).sort_index(); r=p.pct_change()
f=-(r.rolling(10,min_periods=8).std()/r.rolling(60,min_periods=40).std()-1)
for h in [5,10,20,40]:
 vals=[]; ns=[]; dates=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.y).statistic); ns.append(len(z)); dates.append(p.index[i])
 v=np.asarray(vals); print('horizon',h,'dates',len(v),'avgN',round(np.mean(ns),2),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1)*np.sqrt(252),6),'hit',round((v>0).mean(),4))
# selected 20d regime and signal artifact
vals=[]; dates=[]
for i in range(len(p)-20):
 z=pd.concat([f.iloc[i],(p.iloc[i+20]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.y).statistic); dates.append(p.index[i])
v=pd.Series(vals,index=dates)
for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2031-06-11')]:
 q=v.loc[a:b]; print('regime',a,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),6) if len(q)>1 else None)
f.to_csv('scripts/miner_1_20310612_volatility_term_structure_signal.csv',index_label='date')
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'loaded',len(U),'last',END.date())

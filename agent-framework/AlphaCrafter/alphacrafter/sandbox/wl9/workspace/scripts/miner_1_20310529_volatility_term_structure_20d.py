import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2031-05-29')
d={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date'); d[s]=x.loc[:END,'close']
p=pd.DataFrame(d).sort_index(); r=p.pct_change()
# low recent vol relative to long vol, lagged; direction is expected volatility normalization / rebound
f=-(r.rolling(10,min_periods=8).std()/r.rolling(60,min_periods=40).std()-1)
res={}
for h in [5,10,20,40]:
 v=[]; ns=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: v.append(spearmanr(z.iloc[:,0],z.y).statistic); ns.append(len(z))
 v=np.array(v); res[h]=v
 print(h,'dates',len(v),'avgN',round(np.mean(ns),2),'IC',round(np.mean(v),6),'ICIR',round(np.mean(v)/np.std(v,ddof=1)*np.sqrt(252),6),'hit',round(np.mean(v>0),4))
v=[]; ds=[]
for i in range(len(p)-20):
 z=pd.concat([f.iloc[i],(p.iloc[i+20]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.y).statistic);ds.append(p.index[i])
v=pd.Series(v,index=ds)
for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2031-05-29')]:
 q=v.loc[a:b]; print('regime',a,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(252),6))
f.loc[v.index].to_csv('scripts/miner_1_20310529_volatility_term_structure_signal.csv',index_label='date')
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())

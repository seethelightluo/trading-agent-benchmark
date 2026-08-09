import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x
  except: pass
px=pd.DataFrame({s:load(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change(); v=r.rolling(20,min_periods=10).std()
f=-(r.rolling(20,min_periods=15).sum()/v); f=f.clip(lower=f.quantile(.05,axis=1),upper=f.quantile(.95,axis=1),axis=0)
f.insert(0,'date',f.index); f.to_csv('../persistent/factor_signals_miner_1_20270225_volscaled_reversal20.csv',index=False)
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]; ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(d)
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028')]:
  q=np.array([x for x,d in zip(vals,ds) if lo<=str(d)[:4]<=hi]);print(lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('coverage',round(f.iloc[:,1:].notna().sum().sum()/(15*len(f)),4),'period',px.index.min().date(),px.index.max().date())

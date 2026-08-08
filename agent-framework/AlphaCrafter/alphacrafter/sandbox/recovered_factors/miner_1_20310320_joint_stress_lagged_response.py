import pandas as pd, numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv').set_index('date')['close'] for s in S}).sort_index(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv').set_index('date')['close'].reindex(p.index).ffill().pct_change(); d=pd.read_csv('../persistent/index_data/DXY.csv').set_index('date')['close'].reindex(p.index).ffill().pct_change()
zv=(v-v.rolling(60).mean())/v.rolling(60).std(); zd=(d-d.rolling(60).mean())/d.rolling(60).std(); event=((zv>1)&(zd>.25)).astype(float)
# lagged one-day response after joint stress, smoothed over 240 observations
f=r.shift(1).mul(event.shift(1),axis=0).rolling(240).sum().div(event.shift(1).rolling(240).sum(),axis=0); f[event.shift(1).rolling(240).sum()<8]=np.nan; f=f.sub(f.mean(axis=1),axis=0)
print('source_dates',len(p),'instruments',len(S),'event_days',int(event.sum()),'coverage',float(f.notna().mean().mean()),'active_dates',int(f.notna().any(axis=1).sum()))
for h in [1,5,10,20]:
 z=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],(p.shift(-h)/p-1).iloc[i]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z); print('H',h,'dates',len(z),'meanN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
for a,b in [('2020','2023'),('2024','2027'),('2028','2029'),('2030','2031'),('2030-08','2031-03')]:
 vals=[]
 for i in range(len(p)-20):
  if not (str(p.index[i])[:7]>=a and str(p.index[i])[:7]<=b): continue
  q=pd.concat([f.iloc[i],(p.shift(-20)/p-1).iloc[i]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 vals=np.array(vals); print('REG20',a,b,'dates',len(vals),'IC',np.mean(vals) if len(vals) else np.nan,'ICIR',np.mean(vals)/np.std(vals,ddof=1) if len(vals)>1 else np.nan)
rank=f.rank(axis=1,pct=True); print('turnover10',float(rank.diff(10).abs().mean().mean()))
print('LIBRARY_AUDIT: NOT PERFORMED; generic expressions require reconstruction of all admitted historical signals')
# candidate only: within candidate variants, max rho is not a library audit
print('candidate_id joint_stress_lagged_response_240obs')

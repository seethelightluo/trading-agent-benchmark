import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change(); lag=r.shift(1)
vix=pd.read_csv('../persistent/index_data/VIX.csv');vix.date=pd.to_datetime(vix.date);v=vix.set_index('date').close.astype(float).reindex(px.index).ffill()
# Stress reversal: after a lagged 5-session VIX surge, rank assets for reversal of their lagged 5-session move.
vix_surge=v.pct_change(5).shift(1)>v.pct_change(5).rolling(120,min_periods=60).quantile(.60).shift(1)
ret5=lag.rolling(5,min_periods=5).sum(); vol30=lag.rolling(30,min_periods=20).std()
f=(-ret5/vol30).where(vix_surge,0.0)
f=f.sub(f.mean(axis=1),axis=0).replace([np.inf,-np.inf],np.nan)
print('instruments',len(U),'dates',len(px),'coverage',f.notna().mean().mean(),'active',vix_surge.mean())
for h in [1,5,10,20]:
 a=[];ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:a.append(z.x.corr(z.y));ns.append(len(z))
 q=pd.Series(a).dropna(); print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n',len(q),'avgN',np.mean(ns))
a=[]
for i in range(len(px)-1):
 z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+1]/px.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8 and z.x.nunique()>2:a.append((px.index[i],z.x.corr(z.y)))
q=pd.Series(dict(a)); print('daily thirds',*[q.iloc[j*len(q)//3:(j+1)*len(q)//3].mean() for j in range(3)])
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.index=f.index.astype(str);f.to_csv('scripts/miner_3_20331031_vix_stress_reversal_signal.csv')

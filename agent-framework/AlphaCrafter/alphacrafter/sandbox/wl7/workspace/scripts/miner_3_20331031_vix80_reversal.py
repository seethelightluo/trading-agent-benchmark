import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); lag=px.pct_change().shift(1)
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close.astype(float).reindex(px.index).ffill()
vs=v.pct_change(5); gate=vs.shift(1)>vs.rolling(120,min_periods=60).quantile(.8).shift(1)
ret=lag.rolling(5,min_periods=5).sum(); vol=lag.rolling(30,min_periods=20).std(); f=(-ret/vol).where(gate,0.0);f=f.sub(f.mean(axis=1),axis=0)
print('coverage',f.notna().mean().mean(),'active',gate.mean())
for h in [1,5,10,20]:
 a=[];ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:a.append(z.x.corr(z.y));ns.append(len(z))
 q=pd.Series(a).dropna();print(h,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns))
a=[]
for i in range(len(px)-1):
 z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+1]/px.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8:a.append(z.x.corr(z.y))
q=pd.Series(a);print('thirds',[q.iloc[j*len(q)//3:(j+1)*len(q)//3].mean() for j in range(3)],'turn',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.index=f.index.astype(str);f.to_csv('scripts/miner_3_20331031_vix80_reversal_signal.csv')

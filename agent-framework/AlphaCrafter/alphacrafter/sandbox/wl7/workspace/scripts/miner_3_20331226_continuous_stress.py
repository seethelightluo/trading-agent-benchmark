import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill();px=px.loc[px.index<=pd.Timestamp('2033-12-25')];r=px.pct_change(); lag=r.shift(1)
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close.astype(float).reindex(px.index).ffill()
# Continuous stress: positive VIX shock percentile times breadth selloff, applied to normalized 5d reversal.
vs=v.pct_change(5); vp=vs.rolling(120,min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).shift(1)
bread=(-r.rolling(20,min_periods=15).sum()).clip(lower=0).mean(axis=1).shift(1)
stress=vp.clip(lower=0)*bread
f=(-lag.rolling(5,min_periods=5).sum()/lag.rolling(30,min_periods=20).std()).mul(stress,axis=0)
f=f.sub(f.mean(axis=1),axis=0)
print('range',px.index.min().date(),px.index.max().date(),'assets',len(U),'coverage',f.notna().mean().mean(),'active_nonzero',float((stress>0).mean()))
for h in [1,5,10,20]:
 a=[];ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:a.append(z.x.corr(z.y));ns.append(len(z))
 q=pd.Series(a).dropna();print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n_dates',len(q),'avgN',np.mean(ns))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.index=f.index.astype(str);f.to_csv('scripts/miner_3_20331226_continuous_stress_signal.csv')

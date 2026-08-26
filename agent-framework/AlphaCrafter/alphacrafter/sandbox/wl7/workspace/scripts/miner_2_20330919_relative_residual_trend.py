import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change(); lag=px.shift(1)
# Relative residual trend: lagged 40d asset return minus contemporaneous equal-weight
# benchmark return, normalized by trailing 40d idiosyncratic volatility.
m=lag.pct_change(40); bm=m.mean(axis=1); resid=m.sub(bm,axis=0)
vol=r.shift(1).rolling(40).std(); f=(resid/vol).replace([np.inf,-np.inf],np.nan)
print('instruments',len(D),'dates',len(px),'coverage',f.notna().mean().mean())
for h in [1,5,10,20]:
 a=[];ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:a.append(z.x.corr(z.y));ns.append(len(z))
 q=pd.Series(a).dropna();print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n',len(q),'avgN',np.mean(ns))
rank=f.rank(axis=1,pct=True);print('turnover',rank.diff().abs().mean().mean())
f.index=f.index.astype(str);f.to_csv('scripts/miner_2_20330919_relative_residual_trend_signal.csv')

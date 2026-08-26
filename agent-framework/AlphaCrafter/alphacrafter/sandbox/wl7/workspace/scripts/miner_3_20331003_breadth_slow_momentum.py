import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill();r=px.pct_change();
# Slow momentum is active only when lagged breadth confirms a broad advance.
m=px.pct_change(60).shift(1); breadth=(m>0).mean(axis=1); gate=breadth>=0.5
vol=r.rolling(40,min_periods=25).std().shift(1); f=m.div(vol).where(gate,0); f=f.sub(f.mean(axis=1),axis=0)
print('factor=60d vol-scaled momentum gated by lagged breadth >= 50%')
print('instruments',len(U),'dates',len(px),'coverage',f.notna().mean().mean(),'active',gate.mean())
for h in [1,5,10,20]:
 ic=[];ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:ic.append(z.x.corr(z.y));ns.append(len(z))
 q=pd.Series(ic).dropna();print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'dates',len(q),'avgN',np.mean(ns))
q=[]
for i in range(len(px)-1):
 z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+1]/px.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8 and z.x.nunique()>2:q.append(z.x.corr(z.y))
q=pd.Series(q); n=len(q);print('daily thirds',*[q.iloc[j*n//3:(j+1)*n//3].mean() for j in range(3)])
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.index=f.index.astype(str);f.to_csv('scripts/miner_3_20331003_breadth_slow_momentum_signal.csv')

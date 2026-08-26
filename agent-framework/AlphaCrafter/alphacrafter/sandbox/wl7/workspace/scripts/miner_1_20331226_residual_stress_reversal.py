import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); px=px.loc[px.index<=pd.Timestamp('2033-12-25')]
r=px.pct_change(); lag=r.shift(1)
# residual short-term reversal: remove contemporaneous cross-asset 10d common move
m=lag.mean(axis=1)
res10=lag.rolling(10,min_periods=10).sum().sub(m.rolling(10,min_periods=10).sum(),axis=0)
vol=lag.rolling(40,min_periods=30).std()
# activate only in broad weak/high-stress regimes, all inputs lagged
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.astype(float).reindex(px.index).ffill()
vchg=v.pct_change(5); breadth=(lag.rolling(20,min_periods=15).sum()<0).mean(axis=1)
gate=(breadth.shift(1)>.55)&(vchg.shift(1)>vchg.rolling(120,min_periods=60).quantile(.70).shift(1))
f=(-res10/vol).where(gate,0.0); f=f.sub(f.mean(axis=1),axis=0)
print('range',px.index.min().date(),px.index.max().date(),'assets',len(U),'gate_dates',int(gate.sum()),'gate_rate',gate.mean(),'coverage',f.notna().mean().mean())
for h in [1,5,10,20]:
 a=[];ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:a.append(z.x.corr(z.y));ns.append(len(z))
 q=pd.Series(a).dropna(); print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n_dates',len(q),'avgN',np.mean(ns))
q=[]
for i in range(len(px)-1):
 z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+1]/px.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8:q.append(z.x.corr(z.y))
q=pd.Series(q); print('daily thirds',[q.iloc[j*len(q)//3:(j+1)*len(q)//3].mean() for j in range(3)])
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.index=f.index.astype(str); f.to_csv('scripts/miner_1_20331226_residual_stress_reversal_signal.csv')

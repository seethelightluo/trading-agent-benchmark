import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill();px=px.loc[px.index<=pd.Timestamp('2033-12-11')];r=px.pct_change();lag=r.shift(1)
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close.astype(float).reindex(px.index).ffill();vs=v.pct_change(5)
for pct in [.85,.9,.95]:
 gate=(vs.shift(1)>vs.rolling(120,min_periods=60).quantile(pct).shift(1))&((r.rolling(20,min_periods=15).sum()<0).mean(axis=1).shift(1)>.60)
 f=(-lag.rolling(5,min_periods=5).sum()/lag.rolling(30,min_periods=20).std()).where(gate,0);f=f.sub(f.mean(axis=1),axis=0);q=[]
 for i in range(len(px)-1):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+1]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:q.append(z.x.corr(z.y))
 q=pd.Series(q).dropna();print('pct',pct,'active',int(gate.sum()),'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())

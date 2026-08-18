import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill();r=px.pct_change();v=r.rolling(20,min_periods=15).std()*np.sqrt(20)
ret5=np.log(px/px.shift(5));path=(r.rolling(20,min_periods=15).sum().abs()/(r.abs().rolling(20,min_periods=15).sum()+1e-12)).clip(0,1)
q=pd.read_csv('../persistent/index_data/VIX.csv');q['date']=pd.to_datetime(q['date']);vv=q.set_index('date')['close'].reindex(px.index).ffill()
z=(vv-vv.rolling(60,min_periods=30).mean())/(vv.rolling(60,min_periods=30).std()+1e-9);stress=(1+0.6*(z>0).astype(float)).replace(0,np.nan)
f=(-ret5/(v+1e-6))*(0.5+path);f=f.mul(stress,axis=0).shift(1);fr=px.pct_change(10).shift(-10)
ics=[];ns=[];ds=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if np.isfinite(c):ics.append(c);ns.append(len(a));ds.append(dt)
zv=np.array(ics);rank=f.rank(axis=1,pct=True)
print('dates',len(zv),'avgN %.2f'%np.mean(ns),'IC %.8f ICIR %.8f hit %.5f coverage %.5f turnover %.5f'%(zv.mean(),zv.mean()/zv.std(ddof=1),np.mean(zv>0),f.notna().sum(axis=1).mean()/len(U),rank.diff().abs().mean(axis=1).dropna().mean()))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 qv=np.array([x for d,x in zip(ds,ics) if a<=str(d.year)<=b]);print('REG',a,b,len(qv),'IC %.8f ICIR %.8f'%(qv.mean(),qv.mean()/qv.std(ddof=1)))
f.loc[ds].to_csv('scripts/miner_3_20340331_stress_reversal_signal.csv',index_label='date')

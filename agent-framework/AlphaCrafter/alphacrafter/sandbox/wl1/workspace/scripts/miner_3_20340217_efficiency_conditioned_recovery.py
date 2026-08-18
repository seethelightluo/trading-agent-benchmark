import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
# Established recovery-pullback core: favor recent recovery after medium drawdown, risk scaled.
down=r.clip(upper=0).rolling(40,min_periods=20).std()*np.sqrt(40)
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
rec=-(np.log(px/px.shift(60))-0.70*np.log(px/px.shift(10)))/(down+0.5*vol+1e-6)
# Conditioning variable: directional efficiency, magnitude only; low efficiency receives reversal emphasis.
eff=(px.pct_change(20)/(r.abs().rolling(20).sum()+1e-12))/(r.rolling(20).std()+1e-12)
eff=eff.clip(eff.quantile(.05,axis=1),eff.quantile(.95,axis=1),axis=0)
# Novel interpretable interaction: recovery-pullback strengthened when trend is inefficient/noisy.
er=eff.rank(axis=1,pct=True)
f=(rec*(1.5-er)).shift(1)
fr=px.pct_change(10).shift(-10)
ics=[]; ns=[]; dates=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if np.isfinite(c): ics.append(c);ns.append(len(a));dates.append(dt)
z=np.array(ics); print('dates',len(z),'avgN',np.mean(ns),'assets',len(P),'IC %.8f ICIR %.8f hit %.5f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)))
rank=f.rank(axis=1,pct=True); print('coverage %.5f turnover %.5f'%(f.notna().sum(axis=1).mean()/len(U),rank.diff().abs().mean(axis=1).dropna().mean()))
for start,end in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=[v for d,v in zip(dates,ics) if start<=str(d.year)<=end]; q=np.array(q); print('REG',start,end,'n',len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan))
out=f.loc[dates].copy();out.insert(0,'date',out.index);out.to_csv('scripts/miner_3_20340217_efficiency_conditioned_recovery_signal.csv',index=False)
print('range',px.index.min(),px.index.max())

import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-01-23')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Drawdown-recovery velocity: medium trend scaled by distance from trailing peak;
# rewards assets recovering toward highs, but penalizes unresolved drawdowns. Lag all inputs.
r20=px.pct_change(20); peak60=px.rolling(60,min_periods=40).max(); dd=px/peak60-1
vol40=r.rolling(40,min_periods=25).std()*np.sqrt(252)
f=(r20*(1+dd).clip(lower=.05)/(vol40+.05)).shift(1)
print('universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(px.index[i])
 a=np.asarray(I); ds=pd.DatetimeIndex(ds)
 print('h',h,'valid',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,mask in [('2020-25',(ds<pd.Timestamp('2026-01-01'))),('2026+',ds>=pd.Timestamp('2026-01-01')),('2028+',ds>=pd.Timestamp('2028-01-01')),('2029+',ds>=pd.Timestamp('2029-01-01'))]:
  q=a[mask]; print(lab,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'dates',int(mask.sum()))
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
f.to_csv('scripts/miner_1_20300124_drawdown_recovery_velocity_signal.csv',index_label='date')

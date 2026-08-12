import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-05-30')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Cross-sectional acceleration: recent 20-session trend relative to its 60-session baseline,
# residualized cross-sectionally and scaled by realized risk; only lagged data are used.
r20=px.pct_change(20); r60=px.pct_change(60); accel=r20-r60/3
resid=accel.sub(accel.mean(axis=1),axis=0); vol=r.rolling(40,min_periods=20).std(); breadth=r20.gt(0).mean(axis=1)
# broad participation confirms continuation; centered breadth avoids directional market beta
confirm=(0.5+(breadth-0.5).abs()).clip(0.5,1.0)
f=(resid/(vol+0.005)).mul(confirm,axis=0).shift(1)
print('factor xs_acceleration_risk_40d universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(px.index[i])
 a=np.asarray(I); ds=pd.DatetimeIndex(ds); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,mask in [('2020-25',(ds>=pd.Timestamp('2020-01-01'))&(ds<pd.Timestamp('2026-01-01'))),('2026+',ds>=pd.Timestamp('2026-01-01')),('2028+',ds>=pd.Timestamp('2028-01-01')),('2029YTD',ds>=pd.Timestamp('2029-01-01'))]:
  q=a[mask]; print(lab,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'dates',int(mask.sum()))
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum())); f.to_csv('scripts/miner_2_20290531_xs_acceleration_signal.csv',index_label='date')

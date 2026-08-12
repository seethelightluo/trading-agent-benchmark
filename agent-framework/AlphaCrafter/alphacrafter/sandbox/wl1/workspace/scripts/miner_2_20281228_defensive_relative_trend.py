import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-12-27')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()]))
px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change(); mkt=r.mean(axis=1).fillna(0)
win=60; mm=mkt.rolling(win,min_periods=40).mean(); vv=((mkt-mm)**2).rolling(win,min_periods=40).mean()
beta=r.sub(mm,axis=0).mul(mkt-mm,axis=0).rolling(win,min_periods=40).mean().div(vv,axis=0)
ret60=px.pct_change(win); resid=ret60-beta*mkt.rolling(win,min_periods=40).sum().to_numpy()[:,None]
down=r.clip(upper=0).rolling(40,min_periods=25).std()*np.sqrt(252)
peak=px.rolling(120,min_periods=70).max(); dd=(px/peak-1).clip(upper=0)
f=(resid/(1+4*down)*(1+dd)).shift(1).replace([np.inf,-np.inf],np.nan)
print('factor defensive_relative_trend_60 universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(px.index[i])
 a=np.asarray(I); ds=pd.DatetimeIndex(ds)
 def met(q): return (round(q.mean(),6),round(q.mean()/q.std(ddof=1),6)) if len(q)>1 else (np.nan,np.nan)
 print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC/ICIR',met(a),'hit',round(np.mean(a>0),4))
 for lab,mk in [('2020-25',ds<pd.Timestamp('2026-01-01')),('2026+',ds>=pd.Timestamp('2026-01-01')),('2027+',ds>=pd.Timestamp('2027-01-01')),('2028+',ds>=pd.Timestamp('2028-01-01'))]: print(lab,'IC/ICIR',met(a[mk]),'dates',int(mk.sum()))
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
f.to_csv('scripts/miner_2_20281228_defensive_relative_trend_signal.csv',index_label='date')

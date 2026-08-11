import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-09-08')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Candidate: downside-volatility-adjusted breadth. Positive session share is rewarded,
# while negative-session share is penalized more when downside deviation is elevated.
pos=r.gt(0).rolling(60,min_periods=35).mean(); neg=r.lt(0).rolling(60,min_periods=35).mean()
down=np.sqrt((r.clip(upper=0)**2).rolling(60,min_periods=35).mean())
up=np.sqrt((r.clip(lower=0)**2).rolling(60,min_periods=35).mean())
f=((pos-neg)*(up+0.001)/(down+0.001)).shift(1)
print('universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);Ns.append(len(q));ds.append(px.index[i])
 a=np.asarray(I); ds=pd.DatetimeIndex(ds)
 print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for label,mask in [('2026+',ds>=pd.Timestamp('2026-01-01')),('2027',ds>=pd.Timestamp('2027-01-01')),('Q2+',ds>=pd.Timestamp('2027-04-01'))]:
  z=a[mask]; print(label,round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),int(mask.sum()))
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
f.to_csv('scripts/miner_2_20270909_downside_vol_breadth_signal.csv',index_label='date')

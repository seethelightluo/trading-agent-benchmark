import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-09-22')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Drawdown recovery velocity: rebound from trailing 60-session low, scaled by downside risk and confirmed by recent breadth.
low=px.rolling(60,min_periods=40).min(); recovery=px/low-1
down=r.where(r<0,0).pow(2).rolling(40,min_periods=25).mean().pow(.5)
breadth=r.gt(0).rolling(20,min_periods=12).mean()
f=(recovery/(down+0.004)*(0.5+0.5*breadth)).shift(1)
print('universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);Ns.append(len(q));ds.append(px.index[i])
 a=np.asarray(I); ds=pd.DatetimeIndex(ds); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for label,mask in [('2026+',ds>=pd.Timestamp('2026-01-01')),('2027',ds>=pd.Timestamp('2027-01-01')),('2027Q2',ds>=pd.Timestamp('2027-04-01'))]:
  z=a[mask]; print(label,round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),int(mask.sum()))
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
f.to_csv('scripts/miner_3_20270923_drawdown_recovery_velocity_signal.csv',index_label='date')

import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-11-29')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change(); m=r.mean(axis=1); m.name='m'
mu=r.rolling(60,min_periods=40).mean(); mm=m.rolling(60,min_periods=40).mean(); cross=r.mul(m,axis=0).rolling(60,min_periods=40).mean(); den=m.pow(2).rolling(60,min_periods=40).mean()-mm.pow(2); beta=cross.sub(mu.mul(mm,axis=0)).div(den,axis=0)
raw=px.pct_change(60)-beta.mul(m.rolling(60,min_periods=40).sum(),axis=0); vol=r.rolling(40,min_periods=30).std()*np.sqrt(252); f=(raw/(1+vol)).shift(1)
print('factor residual_market_neutral_momentum_60d universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(px.index[i])
 a=np.asarray(I); ds=pd.DatetimeIndex(ds)
 print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,start in [('2020-25','2020-01-01'),('2026+','2026-01-01'),('2027+','2027-01-01'),('2028+','2028-01-01')]:
  q=a[ds>=pd.Timestamp(start)]; print(lab,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'dates',len(q))
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
f.to_csv('scripts/miner_2_20281130_residual_market_neutral_momentum60_signal.csv',index_label='date')

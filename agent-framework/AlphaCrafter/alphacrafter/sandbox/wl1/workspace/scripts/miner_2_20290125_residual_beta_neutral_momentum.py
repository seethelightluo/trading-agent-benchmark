import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-01-24')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change(); m=r.mean(axis=1)
# Residual medium-term momentum: beta-neutralized 30d return versus equal-weight benchmark, stabilized by idiosyncratic volatility.
beta=r.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
res=r.sub(beta.mul(m,axis=0)); resid30=res.rolling(30,min_periods=25).sum(); iv=res.rolling(40,min_periods=25).std()*np.sqrt(252)
f=(resid30/(1+iv)).shift(1).replace([np.inf,-np.inf],np.nan)
print('factor residual_beta_neutral_momentum_30d universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(px.index[i])
 a=np.asarray(I); ds=pd.DatetimeIndex(ds)
 def met(x): return (round(x.mean(),6),round(x.mean()/x.std(ddof=1),6)) if len(x)>1 else (np.nan,np.nan)
 print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'IC/ICIR',met(a),'hit',round(np.mean(a>0),4))
 for lab,mk in [('2020-25',ds<pd.Timestamp('2026-01-01')),('2026+',ds>=pd.Timestamp('2026-01-01')),('2027+',ds>=pd.Timestamp('2027-01-01')),('2028+',ds>=pd.Timestamp('2028-01-01')),('2029+',ds>=pd.Timestamp('2029-01-01'))]: print(lab,met(a[mk]),int(mk.sum()))
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum())); f.to_csv('scripts/miner_2_20290125_residual_beta_neutral_momentum_signal.csv',index_label='date')

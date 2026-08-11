import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.DataFrame(P).ffill(); r=p.pct_change()
asset5=p/p.shift(5)-1; residual=asset5.sub(asset5.mean(axis=1),axis=0)
csvol=r.std(axis=1).rolling(20,min_periods=10).mean(); med=csvol.rolling(120,min_periods=60).median()
regime=(1-2*(csvol>med).astype(float)).shift(1)
f=(-residual).mul(regime,axis=0).shift(1)
lo=f.quantile(.05,axis=1); hi=f.quantile(.95,axis=1); f=f.clip(lo,hi,axis=0)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): I.append(z);N.append(len(q));ds.append(p.index[i])
 a=np.asarray(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 print('annual',h,{y:round(a[[x.year==y for x in ds]].mean(),6) for y in sorted(set(x.year for x in ds))})
rank=f.rank(axis=1,pct=True); print('turnover',round(float((rank-rank.shift(1)).abs().stack().mean()),6),'factor_dates',int(f.notna().sum(axis=1).ge(8).sum()))

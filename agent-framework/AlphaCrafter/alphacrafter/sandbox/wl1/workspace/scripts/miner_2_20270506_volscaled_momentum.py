import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-05-06')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}; ix=sorted(set().union(*[set(x.index) for x in P.values()])); p=pd.DataFrame({s:x.reindex(ix) for s,x in P.items()}).ffill(); r=p.pct_change()
# Medium-horizon volatility-scaled momentum, lagged one day.
f=(r.rolling(60,min_periods=40).sum()/r.rolling(60,min_periods=40).std()).shift(1)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);Ns.append(len(q));ds.append(p.index[i])
 a=np.array(I);print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,5),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10:print('annual10',{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True);print('turnover',round((rank-rank.shift()).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))

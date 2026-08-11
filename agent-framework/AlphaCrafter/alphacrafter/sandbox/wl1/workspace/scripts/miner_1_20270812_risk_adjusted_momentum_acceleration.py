import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-08-12')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
ix=sorted(set().union(*[set(x.index) for x in P.values()]))
p=pd.DataFrame({s:x.reindex(ix) for s,x in P.items()}).ffill(); r=p.pct_change()
# Risk-adjusted momentum acceleration: recent 10d return compared with the
# average daily pace of the prior 30d, normalized by current 20d volatility.
# Signal is lagged one completed day before forward-return measurement.
r10=r.rolling(10,min_periods=8).sum(); r30=r.rolling(30,min_periods=20).sum(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
f=((r10-r30/3)/(vol+1e-8)).shift(1)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[]; Ns=[]; ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): I.append(z);Ns.append(len(q));ds.append(p.index[i])
 a=np.asarray(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==20: print('annual20',{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
for name,start,end in [('recent2026+',pd.Timestamp('2026-01-01'),pd.Timestamp('2027-08-12')),('early',pd.Timestamp('2020-01-01'),pd.Timestamp('2023-12-31'))]:
 vals=[]
 for i in range(len(p)-20):
  if not (start<=p.index[i]<=end): continue
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+20]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.f,q.y).statistic)
 print(name,'dates',len(vals),'IC',round(np.nanmean(vals),6),'ICIR',round(np.nanmean(vals)/np.nanstd(vals,ddof=1),6))

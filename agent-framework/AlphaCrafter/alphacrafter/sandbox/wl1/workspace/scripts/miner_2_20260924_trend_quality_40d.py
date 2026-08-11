import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
idx=sorted(set().union(*[set(v.index) for v in P.values()]))
p=pd.DataFrame({s:v.reindex(idx) for s,v in P.items() for _ in [0]})
r=p.ffill().pct_change()
# Trend-quality factor: medium-horizon return scaled by realized volatility and path efficiency.
ret=p/p.shift(40)-1
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
eff=ret.abs()/(r.abs().rolling(40,min_periods=30).sum()+1e-12)
f=ret/(vol+1e-12)*eff
for h in [5,10,20]:
 I=[]; Ns=[]; dates=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   I.append(spearmanr(q.f,q.y).statistic); Ns.append(len(q)); dates.append(p.index[i])
 a=np.asarray(I)
 print('h',h,'dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10:
  print('annual10', {y:round(a[[d.year==y for d in dates]].mean(),6) for y in sorted(set(d.year for d in dates))})
# rank turnover
z=[]; prev=None
for _,row in f.iterrows():
 q=row.dropna()
 if len(q)>=8:
  rr=q.rank(pct=True)
  if prev is not None:
   ix=rr.index.intersection(prev.index); z.append(np.abs(rr[ix]-prev[ix]).mean())
  prev=rr
print('turnover',round(float(np.mean(z)),6),'factor_dates',int(f.notna().sum(axis=1).ge(8).sum()),'instruments',len(U))
print('corr_raw40',round(float(f.stack().corr(ret.stack())),6))

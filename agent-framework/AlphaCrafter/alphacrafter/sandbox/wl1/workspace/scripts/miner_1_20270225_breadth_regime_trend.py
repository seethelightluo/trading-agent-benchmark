import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()]))
p=pd.DataFrame({s:x.reindex(idx) for s,x in P.items()}).ffill(); r=p.pct_change()
trend=p.pct_change(20)
breadth=(r>0).rolling(10,min_periods=8).mean().mean(axis=1)
regime=np.tanh((breadth-0.5)*6)
f=trend.mul(0.55+0.90*regime,axis=0)
for h in [5,10,20]:
 I=[]; Ns=[]; dates=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   I.append(spearmanr(q.f,q.y).statistic); Ns.append(len(q)); dates.append(p.index[i])
 a=np.array(I)
 print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10: print('annual10',{y:round(a[[d.year==y for d in dates]].mean(),6) for y in sorted(set(d.year for d in dates))})
z=[];prev=None
for _,row in f.iterrows():
 q=row.dropna()
 if len(q)>=8:
  rr=q.rank(pct=True)
  if prev is not None:
   ix=rr.index.intersection(prev.index);z.append(np.abs(rr[ix]-prev[ix]).mean())
  prev=rr
print('turnover_daily',round(np.mean(z),6),'factor_dates',int(f.notna().sum(axis=1).ge(8).sum()),'instruments',15)

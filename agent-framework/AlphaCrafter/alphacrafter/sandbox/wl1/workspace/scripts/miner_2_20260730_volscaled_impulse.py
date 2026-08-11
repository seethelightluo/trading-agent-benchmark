import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut,'close'] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()]));p=pd.DataFrame({s:x.reindex(idx) for s,x in P.items()});f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s,x in P.items():
 r=x.pct_change(); v=r.rolling(20,min_periods=15).std()*np.sqrt(20); f[s]=((x/x.shift(5)-1)/v).reindex(p.index)
for h in [1,5,10,20]:
 vals=[];ns=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: vals.append(spearmanr(q.f,q.y).statistic);ns.append(len(q));ds.append(p.index[i])
 x=np.array(vals);print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
 if h==10: print('annual10d',{int(y):round(x[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
turn=[];prev=None
for _,row in f.iterrows():
 z=row.dropna()
 if len(z)>=8:
  q=z.rank(pct=True)
  if prev is not None:
   ix=q.index.intersection(prev.index);turn.append(np.abs(q[ix]-prev[ix]).mean())
  prev=q
print('turnover',round(np.mean(turn),6),'assets',len(U),'rows',len(p),'coverage_dates',round(f.notna().sum(axis=1).ge(8).mean(),4))

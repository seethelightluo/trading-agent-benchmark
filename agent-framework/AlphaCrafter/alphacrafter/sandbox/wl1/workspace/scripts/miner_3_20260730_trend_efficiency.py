import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); p=pd.DataFrame({s:x.reindex(idx).to_numpy() for s,x in P.items()},index=idx)
f=pd.DataFrame(index=idx,columns=U,dtype=float)
for s in U:
 x=P[s].reindex(idx); r=x.pct_change(); path=r.abs().rolling(20,min_periods=16).sum(); net=x.pct_change(20)
 f[s]=(net/(path+1e-8)).to_numpy()
for h in [5,10,20]:
 ic=[];ns=[];dates=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.y.nunique()>1: ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q));dates.append(f.index[i])
 a=np.array(ic); print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10: print('annual',{int(y):round(a[[d.year==y for d in dates]].mean(),6) for y in sorted(set(d.year for d in dates))})
turn=[];prev=None
for _,row in f.iterrows():
 z=row.dropna()
 if len(z)>=8:
  rank=z.rank(pct=True)
  if prev is not None:
   ix=rank.index.intersection(prev.index);turn.append(np.abs(rank[ix]-prev[ix]).mean())
  prev=rank
print('turnover',round(np.mean(turn),6),'valid_dates',len(turn),'coverage_dates',round(f.notna().sum(axis=1).ge(8).mean(),4))

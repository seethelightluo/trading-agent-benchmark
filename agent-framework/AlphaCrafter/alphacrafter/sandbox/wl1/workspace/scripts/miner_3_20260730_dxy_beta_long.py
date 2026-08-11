import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'
 return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().close.loc[:cut]
x=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=x.pct_change(); d=load('DXY',True).reindex(x.index).ffill(); dr=d.pct_change()
for w in [100,120,150]:
 f=pd.DataFrame(index=x.index,columns=U,dtype=float)
 for i in range(w,len(x)):
  h=r.iloc[i-w:i]; z=dr.iloc[i-w:i]; v=z.var()
  if v>1e-12: f.iloc[i]=h.apply(lambda q:q.cov(z)/v)
 for h in [10]:
  vals=[]; ns=[]; dates=[]
  for i in range(len(x)-h):
   q=pd.concat([f.iloc[i],(x.iloc[i+h]/x.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.y.nunique()>1: vals.append(spearmanr(q.iloc[:,0],q.y).statistic); ns.append(len(q)); dates.append(x.index[i])
  a=np.array(vals); print('w',w,'h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
  print('annual',{int(y):round(a[[d.year==y for d in dates]].mean(),6) for y in sorted(set(d.year for d in dates))})
 # rank turnover
 z=f.rank(axis=1,pct=True); dif=[]
 for i in range(1,len(z)):
  ix=z.iloc[i].dropna().index.intersection(z.iloc[i-1].dropna().index)
  if len(ix)>=8:dif.append(np.abs(z.iloc[i][ix]-z.iloc[i-1][ix]).mean())
 print('turnover',round(np.mean(dif),6),'coverage_dates',round(f.notna().sum(axis=1).ge(8).mean(),4))

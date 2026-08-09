import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
def load(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'
 return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().close.loc[:cut]
# DXY beta: high beta means asset falls with dollar; negative beta is defensive
x=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=x.pct_change()
d=load('DXY',True).reindex(x.index).ffill(); dr=d.pct_change()
for w in [40,60,90]:
 f=pd.DataFrame(index=x.index,columns=U,dtype=float)
 for i in range(w,len(x)):
  h=r.iloc[i-w:i]; z=dr.iloc[i-w:i]; v=z.var()
  if v>1e-12: f.iloc[i]=h.apply(lambda q:q.cov(z)/v)
 for h in [1,5,10]:
  vals=[]; ns=[]
  for s in U:
   ix=x[s].dropna().index
   for j in range(len(ix)-h):
    dt=ix[j]; z=f.loc[dt,s]; y=x.loc[ix[j+h],s]/x.loc[dt,s]-1
    if pd.notna(z) and pd.notna(y): vals.append((dt,s,z,y))
  a=pd.DataFrame(vals,columns=['date','s','f','y']); ic=[]
  for dt,g in a.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1: ic.append(spearmanr(g.f,g.y).statistic)
  q=np.array(ic);print('w',w,'h',h,'dates',len(q),'avgN',round(a.groupby('date').size().mean(),2),'coverage',round(a.groupby('date').size().mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 z=[]
 for dt,g in a[a.date.dt.year>=2020].groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:z.append((dt,spearmanr(g.f,g.y).statistic))
 zz=pd.Series(dict(z)); print('regime', {int(y):round(zz[zz.index.year==y].mean(),5) for y in sorted(zz.index.year.unique())})

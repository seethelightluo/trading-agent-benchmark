import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent/stock_data'
p,fwd={},{}
for s in U:
 d=pd.read_csv(os.path.join(root,s+'.csv')); d.date=pd.to_datetime(d.date); d=d[(d.date>='2020-01-01')&(d.date<='2026-07-15')].sort_values('date'); c=d.set_index('date').close.astype(float); r=c.pct_change(); v=r.rolling(20,min_periods=15).std()*np.sqrt(20); f=-(c/c.shift(3)-1)/v; p[s]=f; fwd[s]=c.shift(-1)/c-1
factor=pd.DataFrame(p); forward=pd.DataFrame(fwd)
for h in [1,5,10]:
 # recompute forward per asset in native trading observations
 ff={}
 for s in U:
  d=pd.read_csv(os.path.join(root,s+'.csv'));d.date=pd.to_datetime(d.date);d=d[(d.date>='2020-01-01')&(d.date<='2026-07-15')].sort_values('date');c=d.set_index('date').close.astype(float);ff[s]=c.shift(-h)/c-1
 y=pd.DataFrame(ff); ics=[];ns=[]
 for dt in factor.index:
  ok=factor.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: ics.append(spearmanr(factor.loc[dt,ok],y.loc[dt,ok]).statistic);ns.append(ok.sum())
 a=np.array(ics);print('H',h,'dates',len(a),'meanIC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'avgN',round(np.mean(ns),2))
r=factor.rank(axis=1,pct=True); print('coverage',round(factor.notna().sum(axis=1).mean()/15,4),'turn',round((r-r.shift()).abs().mean(axis=1).mean(),4))
# pooled correlations
for h in [3,5]:
 z={}
 for s in U:
  d=pd.read_csv(os.path.join(root,s+'.csv'));d.date=pd.to_datetime(d.date);d=d[(d.date>='2020-01-01')&(d.date<='2026-07-15')].sort_values('date');c=d.set_index('date').close.astype(float);z[s]=-(c/c.shift(h)-1)
 z=pd.DataFrame(z); ok=factor.notna()&z.notna(); print('corr rev',h,round(factor.where(ok).stack().corr(z.where(ok).stack()),5))
for a,b in [('2020-01-01','2022-02-28'),('2022-03-01','2024-04-30'),('2024-05-01','2026-07-15')]:
 q=[]
 for dt in factor.loc[a:b].index:
  ok=factor.loc[dt].notna()&forward.loc[dt].notna()
  if ok.sum()>=8:q.append(spearmanr(factor.loc[dt,ok],forward.loc[dt,ok]).statistic)
 q=np.array(q);print('regime',a,len(q),round(q.mean(),5),round(q.mean()/q.std(ddof=1),5))

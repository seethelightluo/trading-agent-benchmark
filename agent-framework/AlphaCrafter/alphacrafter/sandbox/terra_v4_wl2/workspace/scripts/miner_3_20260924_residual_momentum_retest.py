import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-09-23');D={}
for s in U:
 q=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close;D[s]=q[q.index<=cutoff]
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); common=r.median(axis=1,skipna=True)
def roll_beta(x,z,w=60,minn=45):
 ok=x.notna()&z.notna(); xx=x.where(ok); zz=z.where(ok); n=ok.astype(float).rolling(w).sum(); mx=xx.rolling(w).sum()/n; mz=zz.rolling(w).sum()/n
 cov=(xx*zz).rolling(w).sum()/n-mx*mz; vz=(zz*zz).rolling(w).sum()/n-mz*mz
 return (cov/vz).where(n>=minn)
cm20=(1+common).rolling(20,min_periods=20).apply(np.prod,raw=True)-1
for h in [1,5,10]:
 rows=[]
 for s in p.columns:
  f=p[s]/p[s].shift(20)-1-roll_beta(r[s],common)*cm20; y=p[s].shift(-h)/p[s]-1
  for dt in p.index:
   if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]):rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
 a=pd.DataFrame(rows,columns=['date','s','f','y']); ics=[];ds=[];ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ds.append(dt);ns.append(len(g));ics.append(spearmanr(g.f,g.y).statistic)
 z=np.array(ics);print('horizon',h,'dates',len(z),'avg_names',round(np.mean(ns),2),'coverage',round(a.s.nunique()/15,4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
 for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-09-23')]:
  ix=[pd.Timestamp(lo)<=d<=pd.Timestamp(hi) for d in ds];v=z[ix];print(' ',label,'n',len(v),'ICIR',round(v.mean()/v.std(ddof=1),5) if len(v)>1 else None)
 if h==1: print('turnover',round(a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True).diff().abs().mean().mean(),4))

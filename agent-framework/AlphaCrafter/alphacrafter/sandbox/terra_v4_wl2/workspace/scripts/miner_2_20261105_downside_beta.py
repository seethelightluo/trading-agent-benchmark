import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-11-04')
R={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index();d=d[d.index<=cut];R[s]=d.close.pct_change()
bench=pd.concat([R[s] for s in ['SPX','000300.SH','SX5E','N225']],axis=1).mean(axis=1); m=(bench<0).astype(float)
# conditional downside beta, calculated from rolling moments on down-market observations
rows=[]
for s in U:
 r=R[s]; n=m.rolling(60).sum(); ex=(r*m).rolling(60).sum()/n; eb=(bench*m).rolling(60).sum()/n
 cov=((r-ex)*(bench-eb)*m).rolling(60).sum()/n; var=((bench-eb)**2*m).rolling(60).sum()/n
 f=-cov/var; y=r.shift(-1)
 for dt in r.index:
  if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]):rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
a=pd.DataFrame(rows,columns=['date','s','f','y']);ds=[];ics=[];ns=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:ds.append(dt);ns.append(len(g));ics.append(spearmanr(g.f,g.y).statistic)
v=np.array(ics);print('dates',len(v),'avg_names',round(np.mean(ns),2),'coverage',round(a.s.nunique()/15,3),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round(np.mean(v>0),4))
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-11-04')]:
 q=v[[pd.Timestamp(lo)<=d<=pd.Timestamp(hi) for d in ds]];print(label,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),5))
rk=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True);print('turnover',round(rk.diff().abs().mean().mean(),4))
for h in [5,10]:
 z=[]
 for s in U:
  r=R[s]; n=m.rolling(60).sum();ex=(r*m).rolling(60).sum()/n;eb=(bench*m).rolling(60).sum()/n;f=-(((r-ex)*(bench-eb)*m).rolling(60).sum()/n)/(((bench-eb)**2*m).rolling(60).sum()/n);z.append(pd.DataFrame({'f':f,'y':r.shift(-h),'s':s}))
 q=pd.concat(z).reset_index().rename(columns={'index':'date'}).dropna();vv=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:vv.append(spearmanr(g.f,g.y).statistic)
 vv=np.array(vv);print('h',h,'dates',len(vv),'IC',round(vv.mean(),6),'ICIR',round(vv.mean()/vv.std(ddof=1),6))

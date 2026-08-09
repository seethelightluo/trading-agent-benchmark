import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2026-10-07')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); z=macro.reindex(p.index).pct_change()
def beta(x,z,w=60,minn=45):
 ok=x.notna()&z.notna(); n=ok.astype(float).rolling(w).sum(); xx=x.where(ok); zz=z.where(ok)
 mx=xx.rolling(w).sum()/n; mz=zz.rolling(w).sum()/n
 cov=(xx*zz).rolling(w).sum()/n-mx*mz; vz=(zz*zz).rolling(w).sum()/n-mz*mz
 return (cov/vz).where(n>=minn)
rows=[]
for s in p:
 # lower DXY beta expected to outperform when dollar pressure reverses; defensive exposure
 f=-beta(r[s],z)
 for dt in p.index:
  y=p[s].shift(-1)/p[s]-1
  if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]): rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
a=pd.DataFrame(rows,columns=['date','s','f','y']); ics=[];ds=[];ns=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
  ds.append(dt);ns.append(len(g));ics.append(spearmanr(g.f,g.y).statistic)
v=np.array(ics)
print('dates',len(v),'avg_names',round(np.mean(ns),2),'coverage',round(a.s.nunique()/15,4),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round(np.mean(v>0),4))
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-10-07')]:
 q=v[[pd.Timestamp(lo)<=d<=pd.Timestamp(hi) for d in ds]]; print(label,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),5))
r=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean().mean(),4))

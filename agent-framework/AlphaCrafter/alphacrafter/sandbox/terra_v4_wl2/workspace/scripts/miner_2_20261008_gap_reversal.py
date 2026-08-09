import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-10-07')
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); d=d[d.index<=cut]
 # gap is today's open versus prior completed close; signal fades gap, target next close return
 gap=d.open/d.close.shift(1)-1; f=-gap; y=d.close.shift(-1)/d.close-1
 for dt in d.index:
  if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]): rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
a=pd.DataFrame(rows,columns=['date','s','f','y']); ds=[];ics=[];ns=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ds.append(dt);ns.append(len(g));ics.append(spearmanr(g.f,g.y).statistic)
v=np.array(ics); print('dates',len(v),'avg_names',round(np.mean(ns),2),'coverage',round(a.s.nunique()/15,3),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round(np.mean(v>0),4))
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-10-07')]:
 q=v[[pd.Timestamp(lo)<=d<=pd.Timestamp(hi) for d in ds]]; print(label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),5))
r=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean().mean(),4))
for h in [5,10]:
 yy=[];ii=[]
 for s in U:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); gap=d.open/d.close.shift(1)-1; f=-gap; y=d.close.shift(-h)/d.close-1
  for dt in d.index:
   if dt<=cut and pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]): yy.append((dt,s,f.loc[dt],y.loc[dt]))
 b=pd.DataFrame(yy,columns=['date','s','f','y']); q=[]
 for _,g in b.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
 q=np.array(q); print('h',h,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))

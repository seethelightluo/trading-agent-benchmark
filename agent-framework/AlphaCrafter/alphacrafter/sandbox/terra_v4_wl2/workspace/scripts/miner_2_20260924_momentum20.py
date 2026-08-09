import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cutoff=pd.Timestamp('2026-09-23'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); D[s]=x.loc[x.index<=cutoff]
rows=[]
for s,x in D.items():
 f=x.close.pct_change(20); y=x.close.shift(-1)/x.close-1
 for dt in x.index:
  if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]):rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
a=pd.DataFrame(rows,columns=['date','symbol','factor','forward']);z=[];ns=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1:z.append(spearmanr(g.factor,g.forward).statistic);ns.append(len(g))
z=np.array(z);print('dates',len(z),'avg_names',np.mean(ns),'symbols',a.symbol.nunique(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'coverage',a.symbol.nunique()/15)
r=a.assign(rank=a.groupby('date').factor.rank(pct=True)).pivot(index='date',columns='symbol',values='rank');print('turnover',r.diff().abs().mean().mean())
for l,lo,hi in [('2020-22','2020','2022'),('2023-24','2023','2024'),('2025-26','2025','2026')]:
 v=[]
 for dt,g in a.groupby('date'):
  if lo<=str(dt.year)<=hi and len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1:v.append(spearmanr(g.factor,g.forward).statistic)
 v=np.array(v);print(l,len(v),v.mean(),v.mean()/v.std(ddof=1))

import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-09-23')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); D[s]=x.loc[:cutoff]
# Overnight gap reversal: negative prior overnight return, point-in-time at session close.
rows=[]
for s,x in D.items():
 f=-(x.open/x.close.shift(1)-1); y=x.close.shift(-1)/x.close-1
 for dt in x.index:
  if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]): rows.append((dt,s,f.loc[dt],y.loc[dt]))
a=pd.DataFrame(rows,columns=['date','s','f','y']); out=[]; ns=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: out.append(spearmanr(g.f,g.y).statistic); ns.append(len(g))
z=np.array(out); print('dates',len(z),'avg_names',np.mean(ns),'coverage',a.s.nunique()/15); print('daily IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'std',z.std(ddof=1))
for h in [5,10]:
 q=[]
 for s,x in D.items():
  f=-(x.open/x.close.shift(1)-1); y=x.close.shift(-h)/x.close-1
  q += [(dt,s,f.loc[dt],y.loc[dt]) for dt in x.index if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt])]
 q=pd.DataFrame(q,columns=['date','s','f','y']); v=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:v.append(spearmanr(g.f,g.y).statistic)
 v=np.array(v); print(h,'d IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'dates',len(v))
r=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank'); print('turnover',r.diff().abs().mean().mean())
for label,lo,hi in [('2020-22','2020','2022'),('2023-24','2023','2024'),('2025-26','2025','2026')]:
 v=[]
 for dt,g in a.groupby('date'):
  if lo<=str(dt.year)<=hi and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:v.append(spearmanr(g.f,g.y).statistic)
 v=np.array(v); print(label,len(v),v.mean(),v.mean()/v.std(ddof=1))

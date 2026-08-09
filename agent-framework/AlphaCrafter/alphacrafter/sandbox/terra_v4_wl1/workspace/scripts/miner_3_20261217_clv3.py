import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-16')
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); d=d.sort_values('date'); d=d[d.date<=END]
 rg=(d.high-d.low).replace(0,np.nan); d['f']=(-(2*(d.close-d.low)/rg-1)).rolling(3,min_periods=3).mean()
 d['r1']=d.close.pct_change().shift(-1); d['r5']=d.close.shift(-5)/d.close-1; d['r10']=d.close.shift(-10)/d.close-1
 rows += [(z.date,s,z.f,z.r1,z.r5,z.r10) for z in d.itertuples()]
a=pd.DataFrame(rows,columns=['date','s','f','r1','r5','r10']); print('range',a.date.min(),a.date.max())
for h in ['r1','r5','r10']:
 vals=[]
 for dt,g in a.groupby('date'):
  g=g.dropna(subset=['f',h]);
  if len(g)>=8: vals.append(spearmanr(g.f,g[h]).statistic)
 v=np.array(vals); print(h,'dates',len(v),'names',a.groupby('date').f.apply(lambda x:x.notna().sum()).mean(),'mean',v.mean(),'std',v.std(ddof=1),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
print('coverage',a.f.notna().mean())
r=a.dropna(subset=['f']).pivot(index='date',columns='s',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=[]
 for dt,g in a[(a.date>=lo)&(a.date<str(int(hi)+1))].groupby('date'):
  g=g.dropna(subset=['f','r1']);
  if len(g)>=8:z.append(spearmanr(g.f,g.r1).statistic)
 print(lo+'-'+hi,'dates',len(z),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1))

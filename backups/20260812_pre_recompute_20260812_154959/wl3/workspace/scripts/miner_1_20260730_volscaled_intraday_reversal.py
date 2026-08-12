import pandas as pd, numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for s in syms}
rows=[]
for s,x in D.items():
 r=x.close.pct_change(); f=-(x.close/x.open-1)/r.rolling(20,min_periods=15).std().shift(1); y=x.close.shift(-1)/x.close-1
 z=pd.DataFrame({'f':f,'y':y}).dropna().reset_index(); z['sym']=s; rows.append(z)
a=pd.concat(rows,ignore_index=True); nby=a.groupby('date').size(); ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ics.append(spearmanr(g.f,g.y).statistic)
ic=np.array(ics); print('dates',len(ic),'avg_n',nby[nby>=8].mean(),'coverage',len(a)/sum(len(x) for x in D.values()),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=a[(a.date>=lo)&(a.date<=hi)]; v=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:v.append(spearmanr(g.f,g.y).statistic)
 v=np.array(v); print('regime',lo,len(v),v.mean(),v.mean()/v.std(ddof=1))
for h in [1,3,5,10]:
 rows=[]
 for s,x in D.items():
  f=-(x.close/x.open-1)/x.close.pct_change().rolling(20,min_periods=15).std().shift(1); y=x.close.shift(-h)/x.close-1
  z=pd.DataFrame({'f':f,'y':y}).dropna().reset_index(); z['sym']=s; rows.append(z)
 q=pd.concat(rows); v=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:v.append(spearmanr(g.f,g.y).statistic)
 print('decay',h,len(v),np.mean(v))

import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-11-02')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
rows=[]
for s,d in D.items():
 d=d[d.index<=END]; c=d.close; r=c.pct_change(); vol=r.rolling(20).std(); f=-(c.pct_change(3)/(vol*np.sqrt(3)+1e-12))
 for dt in f.index:
  j=d.index.get_loc(dt)
  if j+5>=len(d): continue
  rows.append((dt,s,f.loc[dt],c.iloc[j+5]/c.iloc[j]-1))
x=pd.DataFrame(rows,columns=['date','symbol','f','fr']).dropna(); z=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1: z.append((dt,spearmanr(g.f,g.fr).statistic,len(g)))
a=np.array([q[1] for q in z]); print('dates',len(a),'mean_n',np.mean([q[2] for q in z]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',x.groupby('date').size().mean()/15)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-12-31')]:
 q=[v for dt,v,n in z if str(dt)>=lo and str(dt)<=hi]; print(lo,len(q),np.mean(q) if q else None)

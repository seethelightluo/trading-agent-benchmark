import os,pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-10-20'); ds={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date'); ds[s]=d.close.astype(float)
p=pd.DataFrame(ds).sort_index(); r=p.pct_change(); fac=-(r.rolling(10).std()/r.rolling(30).std()); rows=[]
for dt in fac.index:
 z=pd.DataFrame({'f':fac.loc[dt],'y':p.pct_change(5).shift(-5).loc[dt]}).dropna()
 if len(z)>=8: rows.append((dt,z.f.corr(z.y),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).dropna(); a=x.ic
print('range',p.index.min().date(),p.index.max().date(),'dates',len(x),'avgN',round(x.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'coverage',round(fac.notna().mean().mean(),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
 q=a[(x.date.dt.year>=lo)&(x.date.dt.year<=hi)]; print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)

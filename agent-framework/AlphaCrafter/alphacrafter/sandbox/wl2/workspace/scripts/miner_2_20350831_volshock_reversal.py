import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={s:get_stock_daily_data(s,5000) for s in U}; rows=[]
for s,d in D.items():
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); r=np.log(d.close).diff()
 # shock reversal: fade recent move, amplified when short vol is unusually high
 f=-np.log(d.close/d.close.shift(5))*(r.rolling(5).std()/r.rolling(20).std()); y=np.log(d.close.shift(-10)/d.close)
 rows.append(pd.DataFrame({'date':d.index,'f':f.values,'y':y.values}))
x=pd.concat(rows,ignore_index=True); ics=[]; ns=[]; ds=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.f.nunique()>1: ics.append(g.f.corr(g.y)); ns.append(len(g)); ds.append(dt)
i=pd.Series(ics); print('candidate=vol_shock_reversal_5d horizon=10'); print('dates',len(i),'avg',np.mean(ns),'coverage',len(x.dropna())/(len(x.date.unique())*15)); print('IC',i.mean(),'ICIR',i.mean()/i.std(ddof=1),'hit',np.mean(i>0))
yr=pd.Series(ds).astype(str).str[:4]
for lab,m in [('20-22',yr.isin(['2020','2021','2022'])),('23-25',yr.isin(['2023','2024','2025'])),('26-29',yr.isin(['2026','2027','2028','2029'])),('30-35',yr.isin(['2030','2031','2032','2033','2034','2035']))]:
 a=i[m.values]; print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1))
print('last',x.date.max())

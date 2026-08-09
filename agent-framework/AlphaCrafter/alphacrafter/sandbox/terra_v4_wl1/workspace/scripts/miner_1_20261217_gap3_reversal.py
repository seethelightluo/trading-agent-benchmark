import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-16'); rows=[]
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv'); p.date=pd.to_datetime(p.date); p=p[p.date<=END].sort_values('date').copy(); p['factor']=-(p.open/p.close.shift(1)-1).rolling(3,min_periods=3).mean(); p['fwd']=p.close.pct_change().shift(-1); rows += p[['date','factor','fwd']].assign(symbol=s).to_dict('records')
o=pd.DataFrame(rows).dropna(); dates=[]; vals=[]; ns=[]
for d,g in o.groupby('date'):
 if len(g)>=8:
  c=g.factor.rank().corr(g.fwd.rank())
  if np.isfinite(c): dates.append(d); vals.append(c); ns.append(len(g))
a=np.array(vals); print('dates',len(a),'avgN',np.mean(ns),'coverage',len(o)/(15*o.date.nunique()),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'turnover',o.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean().mean())
for name,years in [('2020-22',(2020,2021,2022)),('2023-24',(2023,2024)),('2025-26',(2025,2026))]:
 z=a[[d.year in years for d in dates]]; print(name,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
o.assign(date=o.date.dt.strftime('%Y-%m-%d'))[['date','symbol','factor']].to_csv('scripts/miner_1_20261217_gap3_signal.csv',index=False)

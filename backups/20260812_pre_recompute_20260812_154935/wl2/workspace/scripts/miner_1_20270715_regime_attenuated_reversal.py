import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}; D={s:d.set_index('date').sort_index() for s,d in D.items() if d is not None and len(d)>100}
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(20).std(); trend=c.pct_change(20)
 # short-term reversal, attenuated in strong trends and scaled by recent risk
 f=(-r/(.001+vol))*(1-(trend.abs()/(trend.abs()+.05)))
 for dt in d.index: rows.append((dt,s,f.get(dt),r.shift(-1).get(dt)))
x=pd.DataFrame(rows,columns=['date','s','f','y']); obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8: obs.append((dt,len(g),g.f.corr(g.y)))
o=pd.DataFrame(obs,columns=['date','n','ic']).set_index('date'); mu=o.ic.mean(); sd=o.ic.std(ddof=1)
print('range',o.index.min(),o.index.max(),'dates',len(o),'avgN',o.n.mean(),'coverage',o.n.sum()/(len(o)*15)); print('IC %.8f ICIR %.8f hit %.4f'%(mu,mu/sd,(o.ic>0).mean()))
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 z=o.loc[a:b].ic; print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1))
for h in [3,5,10]:
 q=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(20).std(); trend=c.pct_change(20); f=(-r/(.001+vol))*(1-trend.abs()/(trend.abs()+.05)); y=c.pct_change(h).shift(-h); q += [(dt,s,f.get(dt),y.get(dt)) for dt in d.index]
 q=pd.DataFrame(q,columns=['date','s','f','y']); z=[]
 for dt,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8:z.append(g.f.corr(g.y))
 z=pd.Series(z).dropna(); print('decay',h,z.mean(),z.mean()/z.std(ddof=1))
print('turnover_proxy',x.dropna().sort_values(['s','date']).groupby('s').f.apply(lambda z:(z.diff().abs()>0).mean()).mean())

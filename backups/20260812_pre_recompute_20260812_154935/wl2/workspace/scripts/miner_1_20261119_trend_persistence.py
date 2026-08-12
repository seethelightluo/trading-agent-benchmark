import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
D={s:d.set_index('date').sort_index() for s,d in D.items() if d is not None and len(d)>100}
# relative trend persistence: 20d return, rewarded only when 5d and 20d direction agree; risk scale 60d vol
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change()
 sig=(c.pct_change(20)/(0.01+np.sqrt(20)*r.rolling(60).std()))
 agree=(np.sign(c.pct_change(5))==np.sign(c.pct_change(20))).astype(float)
 sig=sig*agree
 for dt in d.index:
  rows.append((dt,s,sig.get(dt),c.pct_change().shift(-1).get(dt)))
x=pd.DataFrame(rows,columns=['date','s','f','y'])
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8: obs.append((dt,len(g),g.f.corr(g.y)))
o=pd.DataFrame(obs,columns=['date','n','ic']).set_index('date')
print('range',o.index.min(),o.index.max(),'dates',len(o),'avgN',o.n.mean(),'coverage',o.n.sum()/(len(o)*15))
mu=o.ic.mean(); sd=o.ic.std(ddof=1)
print('IC %.8f ICIR %.8f hit %.4f'%(mu,mu/sd, (o.ic>0).mean()))
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026-01-01','2026-11-18')]:
 z=o.loc[a:b].ic; print(a,b,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
for h in [3,5,10]:
 rr=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change(); f=(c.pct_change(20)/(0.01+np.sqrt(20)*r.rolling(60).std()))*((np.sign(c.pct_change(5))==np.sign(c.pct_change(20))).astype(float))
  y=c.pct_change(h).shift(-h)
  rr += [(dt,s,f.get(dt),y.get(dt)) for dt in d.index]
 q=pd.DataFrame(rr,columns=['date','s','f','y']); z=[]
 for dt,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8:z.append(g.f.corr(g.y))
 z=pd.Series(z).dropna(); print('decay',h,z.mean(),z.mean()/z.std(ddof=1))
print('turnover proxy',x.dropna().sort_values(['s','date']).groupby('s').f.apply(lambda z:(z.diff().abs()>0).mean()).mean())

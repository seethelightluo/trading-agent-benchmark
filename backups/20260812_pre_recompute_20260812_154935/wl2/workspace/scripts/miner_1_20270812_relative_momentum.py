import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
D={s:d.set_index('date').sort_index() for s,d in D.items() if d is not None and len(d)>100}
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); v=r.rolling(20).std()
 m20=c.pct_change(20); m5=c.pct_change(5)
 # Relative medium-term momentum, volatility normalized, with short trend agreement
 rows += [(dt,s,m20.get(dt),m5.get(dt),v.get(dt),r.shift(-1).get(dt)) for dt in d.index]
x=pd.DataFrame(rows,columns=['date','s','m20','m5','vol','y'])
# cross-sectional median is formed date by date, using only current/past prices
x['med20']=x.groupby('date').m20.transform('median')
x['f']=(x.m20-x.med20)/(0.001+x.vol) * np.sign(x.m5)
obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8: obs.append((dt,len(g),g.f.corr(g.y)))
o=pd.DataFrame(obs,columns=['date','n','ic']).set_index('date'); mu=o.ic.mean(); sd=o.ic.std(ddof=1)
print('range',o.index.min(),o.index.max(),'dates',len(o),'avgN',round(o.n.mean(),2),'coverage',round(o.n.sum()/(len(o)*15),4))
print('IC %.8f ICIR %.8f hit %.4f'%(mu,mu/sd,(o.ic>0).mean()))
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 z=o.loc[a:b].ic; print('regime',a,b,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for h in [3,5,10]:
 rows2=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change(); v=r.rolling(20).std(); m20=c.pct_change(20); m5=c.pct_change(5)
  rows2 += [(dt,s,m20.get(dt),m5.get(dt),v.get(dt),c.pct_change(h).shift(-h).get(dt)) for dt in d.index]
 q=pd.DataFrame(rows2,columns=['date','s','m20','m5','vol','y']); q['med20']=q.groupby('date').m20.transform('median'); q['f']=(q.m20-q.med20)/(.001+q.vol)*np.sign(q.m5)
 z=[]
 for dt,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8:z.append(g.f.corr(g.y))
 z=pd.Series(z).dropna(); print('decay',h,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
# turnover as rank movement proxy, and raw signal change
z=x.dropna().sort_values(['s','date']).groupby('s').f.apply(lambda q:q.diff().abs().mean()).mean()
print('turnover_abs_change_proxy',round(z,6))
print('instruments',len(D))

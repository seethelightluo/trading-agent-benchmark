import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U};D={s:d.set_index('date').sort_index() for s,d in D.items() if d is not None and len(d)>100}
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); down=r.where(r<0,0.0)
 # medium horizon reward relative to realized downside risk, with a mild short-term confirmation
 dd=np.sqrt((down**2).rolling(30,min_periods=20).mean())
 f=(c.pct_change(10)/(0.001+dd))*(0.5+0.5*np.sign(c.pct_change(3)))
 y=r.shift(-1)
 for dt in d.index: rows.append((dt,s,f.get(dt),y.get(dt)))
x=pd.DataFrame(rows,columns=['date','s','f','y']);obs=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8: obs.append((dt,len(g),g.f.corr(g.y)))
o=pd.DataFrame(obs,columns=['date','n','ic']).set_index('date');m=o.ic.mean();q=o.ic.std(ddof=1)
print('range',o.index.min(),o.index.max(),'dates',len(o),'avgN',round(o.n.mean(),2),'coverage',round(o.n.sum()/(len(o)*15),4));print('IC %.8f ICIR %.8f hit %.4f'%(m,m/q,(o.ic>0).mean()))
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 z=o.loc[a:b].ic;print('regime',a,b,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for h in [3,5,10]:
 z=[]
 for s,d in D.items():
  c=d.close.astype(float);r=c.pct_change();down=r.where(r<0,0.0);dd=np.sqrt((down**2).rolling(30,min_periods=20).mean());f=(c.pct_change(10)/(0.001+dd))*(0.5+0.5*np.sign(c.pct_change(3)));y=c.pct_change(h).shift(-h)
  for dt in d.index:z.append((dt,f.get(dt),y.get(dt)))
 z=pd.DataFrame(z,columns=['date','f','y']);ic=[]
 for dt,g in z.groupby('date'):
  g=g.dropna()
  if len(g)>=8:ic.append(g.f.corr(g.y))
 ic=pd.Series(ic).dropna();print('decay',h,'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6))
print('instruments',len(D))

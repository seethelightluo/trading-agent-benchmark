import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def f(s):
 for g in (get_index_daily_data,get_stock_daily_data):
  try:
   x=g(s,days=4000)
   if x is not None and len(x): return x
  except: pass
D={s:f(s) for s in U}; P=pd.DataFrame({s:x.set_index('date').close for s,x in D.items() if x is not None}).sort_index().ffill(); R=P.pct_change()
# dispersion of 5d rolling asset returns; activate reversal only in high-dispersion dates
r5=P.pct_change(5); cs=r5.std(axis=1); gate=cs>cs.rolling(60).median(); vol=R.rolling(20).std()*np.sqrt(20); F=-(r5)/vol; F=F.where(gate, np.nan); FW=P.shift(-10)/P-1
rows=[]
for d in P.index:
 z=pd.concat([F.loc[d],FW.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['d','ic','n']).set_index('d')
def o(label,q):
 q=q.dropna(); print(label,'dates',len(q),'mean_n',round(r.loc[q.index].n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
o('full',r.ic)
for a,b in [('2020','2023-12-31'),('2024','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31'),('2028-04-05','2029-04-04')]:o(a,r.loc[a:b].ic)
print('coverage_universe',len(D)/15)

import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=3600)
    if x is not None and len(x):
        x=x.sort_values('date').drop_duplicates('date').set_index('date')
        D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
ret60=p.pct_change(60); eff=ret60.abs()/(r.abs().rolling(60).sum()+1e-12); fac=ret60*eff
rows=[]
for dt in fac.index:
    z=pd.concat([fac.loc[dt],(p.shift(-10).div(p)-1).loc[dt]],axis=1).dropna()
    if len(z)>=8: rows.append((dt,float(z.iloc[:,0].corr(z.iloc[:,1])),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stat(x): return (round(x.mean(),6),round(x.mean()/x.std(ddof=1)*np.sqrt(len(x)),4),round((x>0).mean(),4),len(x))
print('range',q.index.min(),q.index.max(),'dates',len(q),'meanN',q.n.mean(),'coverage',q.n.mean()/15)
for name,x in [('full',q.ic),('2020-23',q.loc[:'2023-12-31'].ic),('2024-26',q.loc['2024-01-01':'2026-12-31'].ic),('2027-28',q.loc['2027-01-01':'2028-12-31'].ic),('2029',q.loc['2029-01-01':].ic),('recent252',q.ic.tail(252))]: print(name,stat(x))
for h in [5,10,20]:
 ff=p.shift(-h).div(p)-1; rr=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('horizon',h,stat(pd.Series(rr)))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())

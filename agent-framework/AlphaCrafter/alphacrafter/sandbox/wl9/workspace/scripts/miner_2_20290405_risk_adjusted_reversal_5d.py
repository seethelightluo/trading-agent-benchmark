import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            x=f(s, days=4000)
            if x is not None and len(x): return x
        except Exception: pass
    return None
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
# align close by date
px=pd.DataFrame({s:x.set_index('date').close for s,x in D.items()}).sort_index().ffill()
ret=px.pct_change()
# factor: fade recent 5d move, scaled by trailing 20d realized vol; only after sufficient history
fac=-(px/px.shift(5)-1)/(ret.rolling(20).std()*np.sqrt(20))
fwd=px.shift(-10)/px-1
rows=[]
for d in px.index:
    a=fac.loc[d]; b=fwd.loc[d]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def out(name,x):
    x=x.dropna(); print(name,'dates',len(x),'mean_n',round(r.loc[x.index,'n'].mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
out('full',r.ic)
for a,b in [('2020','2023-12-31'),('2024','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31'),('2028-04-05','2029-04-04')]: out(a,r.loc[a:b].ic)
print('coverage',round(len(D)/15,3),'last',px.index[-1].date())
# decay
for h in [1,5,10,20]:
 f=px.pct_change(h)*-1/(ret.rolling(20).std()*np.sqrt(20)); fw=px.shift(-h)/px-1; q=[]
 for d in px.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q).dropna(); print('decay',h,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))

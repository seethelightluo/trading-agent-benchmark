import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    try: d=get_index_daily_data(s, days=4000)
    except Exception: d=None
    if d is None or len(d)<100:
        try: d=get_stock_daily_data(s, days=4000)
        except Exception: d=None
    return d
xs={}
for s in U:
    d=fetch(s)
    if d is not None and len(d): xs[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(xs).sort_index().loc[:pd.Timestamp('2030-09-18')]
r=np.log(p/p.shift(1)); ret=p/p.shift(40)-1; med=ret.median(axis=1)
f=ret.sub(med,axis=0)/(r.rolling(40).std()*np.sqrt(252))
for h in [5,10,20]:
    fr=p.shift(-h)/p-1; vals=[]; dates=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
    q=pd.Series(vals,index=dates).dropna()
    print('H',h,'dates',len(q),'meanN',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
    for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-09-18')]:
        z=q.loc[a:b]; print(a,round(z.mean(),5),round(z.mean()/z.std(ddof=1),5),len(z))
rank=f.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean().mean(),6),'assets',len(xs),'dates',len(p))

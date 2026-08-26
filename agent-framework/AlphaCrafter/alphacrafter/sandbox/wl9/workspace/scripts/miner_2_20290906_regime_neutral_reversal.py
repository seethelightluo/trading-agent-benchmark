import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s,days=3600)
    except Exception:
      try: x=get_index_daily_data(s,days=3600)
      except Exception: x=None
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').set_index('date')
        D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
cs=r.sub(r.median(axis=1),axis=0); vol=r.rolling(20).std()
f=-(cs.rolling(5).sum()/vol.rolling(5).mean())
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 z=pd.Series(vals).dropna(); ic=z.mean(); ir=ic/z.std(ddof=1)*np.sqrt(len(z))
 print(h,'dates',len(z),'assets',len(p.columns),'IC',round(ic,8),'ICIR',round(ir,8),'hit',round((z>0).mean(),4))
fr=p.shift(-10)/p-1; z=[]; dates=[]
for d in f.index:
 a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(a)>=8: z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(d)
z=pd.Series(z,index=pd.to_datetime(dates))
for name,lo,hi in [('recent252',z.index[-252],z.index[-1]),('2029','2029-01-01','2029-09-05'),('2020-23','2020-01-01','2023-12-31'),('2024-26','2024-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31')]:
 q=z.loc[lo:hi]; print(name,'n',len(q),'IC',round(q.mean(),8),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),8) if len(q)>1 else None)
print('coverage',round(f.notna().sum(axis=1).mean()/15,4),'obs',len(z))

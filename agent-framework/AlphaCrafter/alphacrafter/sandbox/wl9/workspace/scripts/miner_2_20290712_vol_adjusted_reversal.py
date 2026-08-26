import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=4000)
    if x is not None and len(x):
        x=x.copy(); x.date=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').sort_index()
        D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
sig=-(p.pct_change(10))/(r.rolling(30).std()*np.sqrt(10)); fwd=p.shift(-10)/p-1
rows=[]; ns=[]; dates=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(d)
ic=pd.Series(rows,index=dates).dropna(); print('dates',len(ic),'mean_n',np.mean(ns),'coverage',len(ic)/len(p))
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'start',ic.index.min(),'end',ic.index.max())
for label,mask in [('2020-23',ic.index<'2024-01-01'),('2024-26',(ic.index>='2024-01-01')&(ic.index<'2027-01-01')),('2027+',ic.index>='2027-01-01'),('recent252',ic.index>=ic.index.max()-pd.Timedelta(days=365))]:
 q=ic[mask]; print(label,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())

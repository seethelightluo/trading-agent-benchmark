import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=4000)
    if x is not None and len(x):
        z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date)
        D[s]=z.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); bench=r.mean(axis=1); rows=[]
for i,t in enumerate(p.index):
    if i<125 or i+10>=len(p): continue
    ret20=p.iloc[i]/p.iloc[i-20]-1; vol20=r.iloc[i-19:i+1].std().replace(0,np.nan); b20=bench.iloc[i-19:i+1].sum()
    sig=(ret20-b20)/vol20 + (0.35 if b20<0 else -0.10)*ret20
    for h in [1,5,10]:
        f=p.shift(-h).iloc[i]/p.iloc[i]-1; q=pd.concat([sig,f],axis=1).dropna(); q.columns=['s','f']
        if len(q)>=8: rows.append((t,h,len(q),q.s.rank().corr(q.f.rank())))
A=pd.DataFrame(rows,columns=['date','h','n','ic']); print('range',p.index.min().date(),p.index.max().date(),'assets',len(D),'rows',len(A))
for h in [1,5,10]:
 q=A[A.h==h]; x=q.ic
 print('H',h,'dates',len(q),'mean_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for label,cond in [('recent',q.date>=pd.Timestamp('2027-07-01')),('online',q.date>=pd.Timestamp('2026-07-16')),('ytd',q.date>=pd.Timestamp('2028-01-01'))]:
  y=q[cond].ic; print(label,len(y),round(y.mean(),6),round(y.mean()/y.std(ddof=1),6) if len(y)>1 else np.nan)

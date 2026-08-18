import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
    z=get_stock_daily_data(s,days=3000)
    if z is not None and len(z):
        z=z.copy(); z.date=pd.to_datetime(z.date); P[s]=z.set_index('date').close.astype(float)
dates=sorted(set.intersection(*[set(v.index) for v in P.values()]))
rows=[]
for i,d in enumerate(dates):
    if i<41 or i+10>=len(dates): continue
    f={}; y={}
    for s,x in P.items():
        if not all(k in x.index for k in (dates[i-40],dates[i-20],dates[i-10],dates[i+10])): continue
        rr=x.pct_change().loc[:d].tail(10).dropna()
        # medium momentum, conditioned on consistency of recent daily direction
        f[s]=(x.loc[d]/x.loc[dates[i-20]]-1) * (0.5+0.5*(rr>0).mean())
        y[s]=x.loc[dates[i+10]]/x.loc[d]-1
    if len(f)>=8:
        ic=pd.Series(f).corr(pd.Series(y),method='spearman')
        if np.isfinite(ic): rows.append((d,ic,len(f)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(15*len(q)))
print('ic',q.ic.mean(),'icir',q.ic.mean()/q.ic.std(ddof=1),'hit', (q.ic>0).mean())
for a,b in [('2020-2021','2021-12-31'),('2022-2023','2023-12-31'),('2024-2025','2025-12-31'),('2026-2028','2028-02-24')]:
    st={'2020-2021':'2020-01-01','2022-2023':'2022-01-01','2024-2025':'2024-01-01','2026-2028':'2026-01-01'}[a]
    z=q.loc[st:b].ic
    print(a,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan,'hit',(z>0).mean())

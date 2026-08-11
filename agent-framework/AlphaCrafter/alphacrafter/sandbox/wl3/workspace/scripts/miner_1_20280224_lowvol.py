import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    try: d=get_stock_daily_data(s, days=4000)
    except Exception as e: print('skip',s,e); continue
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); frames[s]=d.set_index('date').sort_index()
print('symbols',len(frames),'range',min(x.index.min() for x in frames.values()),max(x.index.max() for x in frames.values()))
rows=[]
for s,d in frames.items():
    px=d['close'].astype(float); r=px.pct_change()
    sig=-(r.rolling(20,min_periods=15).std()*np.sqrt(252)).shift(1)
    for h in [1,3,5,10]:
        z=pd.DataFrame({'f':sig,'y':px.shift(-h)/px-1}).dropna(); z['date']=z.index; z['symbol']=s; z['h']=h; rows.append(z)
x=pd.concat(rows,ignore_index=True)
for h in [1,3,5,10]:
    vals=[]
    for dt,g in x[x.h==h].groupby('date'):
        if len(g)>=8: vals.append(g.f.corr(g.y,method='spearman'))
    ic=pd.Series(vals).dropna(); print('H',h,'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'n',len(ic))
a=x[x.h==5]; wide=a.pivot(index='date',columns='symbol',values='f').rank(axis=1,pct=True); print('turnover',wide.diff().abs().mean().mean(),'rows',len(a))
icrows=[]
for dt,g in a.groupby('date'):
 if len(g)>=8: icrows.append((dt,g.f.corr(g.y,method='spearman')))
ic=pd.Series(dict(icrows))
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026+','2026-01-01','2099-01-01')]:
 q=ic.loc[lo:hi]; print(label,q.mean(),q.mean()/q.std(ddof=1),len(q),(q>0).mean())
print('recent120',ic.tail(120).mean(),ic.tail(120).mean()/ic.tail(120).std(ddof=1),len(ic.tail(120)),(ic.tail(120)>0).mean(),'cutoff',ic.index.max())

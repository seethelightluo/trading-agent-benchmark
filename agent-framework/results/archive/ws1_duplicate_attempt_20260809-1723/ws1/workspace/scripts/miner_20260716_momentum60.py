import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
    try: D[a]=get_stock_daily_data(a,days=2000)
    except Exception:
        try: D[a]=get_index_daily_data(a,days=2000)
        except Exception: D[a]=None
P=pd.concat([d.set_index('date').close.astype(float).rename(a) for a,d in D.items() if d is not None and len(d)>200],axis=1).sort_index()
R=P.pct_change()
# trend factor: prior 60d return, evaluated 1,5,10 day forward
for h in [1,5,10]:
    vals=[]; dates=[]
    for i in range(80,len(P)-h):
        x=(P.iloc[i-1]/P.iloc[i-61]-1); y=P.iloc[i+h]/P.iloc[i]-1
        z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
        if len(z)>=8: vals.append(z.x.corr(z.y)); dates.append(P.index[i])
    q=pd.Series(vals,index=dates).dropna(); print('N',len(P.columns),'dates',len(q),'h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit', (q>0).mean())
# turnover of rank direction / cross section
x=P.shift(1)/P.shift(61)-1
rank=x.rank(axis=1,pct=True)
print('coverage',x.notna().sum(axis=1).ge(8).mean(),'turnover',rank.diff().abs().mean(axis=1).mean())
print('range',P.index.min(),P.index.max())

import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(sym):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            x=f(sym, days=5000)
            if x is not None and len(x)>0:return x
        except Exception: pass
    return None
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
print('assets',len(D),{s:len(x) for s,x in D.items()})
px=pd.concat({s:x.set_index(pd.to_datetime(x.date)).close for s,x in D.items()},axis=1).sort_index().ffill()
# one interpretable idea: 60d continuation scaled by slower 60d realized volatility
ret=px.pct_change(60); vol=px.pct_change().rolling(60).std()*np.sqrt(60); fac=ret/vol.replace(0,np.nan)
rows=[]
for h in [5,10,20]:
  ic=[]; n=[]; dates=[]
  fwd=px.shift(-h)/px-1
  for dt in fac.index:
    a=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
    if len(a)>=8 and a.iloc[:,0].nunique()>1 and a.iloc[:,1].nunique()>1:
      ic.append(a.iloc[:,0].rank().corr(a.iloc[:,1].rank())); n.append(len(a)); dates.append(dt)
  z=pd.Series(ic,index=dates).dropna(); mean=z.mean(); sd=z.std(ddof=1)
  print('horizon',h,'dates',len(z),'avg_n',np.mean(n),'IC',mean,'ICIR',mean/sd if sd else np.nan,'hit',(z>0).mean(),'coverage',fac.loc[dates].notna().mean().mean())
print('cutoff',px.index.max().date())

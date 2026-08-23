import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s,days=2200)
            if x is not None and len(x)>0:return x
        except Exception: pass
    return None
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
# Align by common calendar dates; factor is lagged one day, forward return h days
px=pd.DataFrame({s:x.set_index('date')['close'] for s,x in D.items()}).sort_index()
r=px.pct_change()
# acceleration: medium trend minus short trend, positive means recent trend improving
f=(px.pct_change(20)-px.pct_change(5)).shift(1)
for h in [1,3,5,10]:
    vals=[]
    for dt in f.index:
        a=f.loc[dt]; y=px.pct_change(h).shift(-h).loc[dt]
        z=pd.concat([a,y],axis=1).dropna()
        if len(z)>=8:
            vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
    q=pd.DataFrame(vals,columns=['date','ic','n'])
    print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2) if len(q) else 0,'coverage',round(q.n.mean()/len(U),4) if len(q) else 0,'IC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5),'hit',round((q.ic>0).mean(),4))
    if len(q):
      for name,mask in [('2025-26',(q.date>='2025-01-01')&(q.date<'2027-01-01')),('2027YTD',q.date>='2027-01-01')]:
       z=q[mask]; print(name,len(z),round(z.ic.mean(),5) if len(z) else None,round(z.ic.mean()/z.ic.std(ddof=1),5) if len(z)>1 else None)
# turnover daily rank changes
ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean().mean(),'period',px.index.min(),px.index.max(),'assets',len(D))

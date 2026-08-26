import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
    d=None
    try: d=get_index_daily_data(s,days=4000)
    except Exception: pass
    if d is None or len(d)<250:
      try: d=get_stock_daily_data(s,days=4000)
      except Exception: d=None
    if d is not None and len(d)>250: xs[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(xs).sort_index().ffill(); r=p.pct_change()
# Compression-confirmed trend: lagged 20d return, amplified when recent volatility is below its 60d baseline.
vol20=r.rolling(20).std(); vol60=r.rolling(60).std()
compression=(vol60/vol20).clip(0.5,2.0)
f=p.pct_change(20)*compression/(vol20*np.sqrt(252)).replace(0,np.nan)
rows=[]
for i in range(len(p)-10):
 q=pd.concat([f.iloc[i],(p.shift(-10).iloc[i]/p.iloc[i]-1)],axis=1).dropna()
 if len(q)>=8: rows.append((p.index[i],len(q),q.iloc[:,0].corr(q.iloc[:,1],method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); a=x.ic.dropna()
print('assets',len(xs),'dates',len(x),'mean_n',x.n.mean(),'coverage',len(x)/(len(p)-10))
print('IC',a.mean(),'std',a.std(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'recent180',a.tail(180).mean(),'recent500',a.tail(500).mean())
for h in [1,5,10,20]:
 rr=p.shift(-h)/p-1; vals=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],rr.iloc[i]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 z=pd.Series(vals).dropna(); print('H',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std())
print('turnover',f.rank(pct=True).diff().abs().mean().mean())
# regime blocks
for lo,hi in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034')]:
 z=a.loc[lo:hi]; print(lo,hi,'n',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std() if z.std()>0 else np.nan)

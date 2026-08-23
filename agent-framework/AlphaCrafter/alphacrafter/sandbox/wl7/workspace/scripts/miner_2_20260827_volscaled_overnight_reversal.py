import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
rows=[]
for s,d in D.items():
 x=d[['open','close']].replace([np.inf,-np.inf],np.nan)
 gap=x.open/x.close.shift(1)-1
 vol=d.close.pct_change().rolling(20).std()
 f=(-gap/vol).replace([np.inf,-np.inf],np.nan)
 for h in [1,5,10]:
  r=d.close.shift(-h)/d.close-1
  z=pd.DataFrame({'f':f,'r':r}).dropna(); z['asset']=s; z['h']=h; rows.append(z.reset_index())
R=pd.concat(rows); print('assets',len(D),'period',R.date.min(),R.date.max())
for h in [1,5,10]:
 q=R[R.h==h]; out=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: out.append(g.f.corr(g.r,method='spearman'))
 a=pd.Series(out).dropna(); print('h',h,'dates',len(a),'avg_n',q.groupby('date').size().mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'hit',(a>0).mean())
f=R[R.h==1].pivot(index='date',columns='asset',values='f'); print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=R[(R.h==1)&(R.date.dt.strftime('%Y').between(a,b))]; out=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: out.append(g.f.corr(g.r,method='spearman'))
 x=pd.Series(out); print('regime',a,b,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252))

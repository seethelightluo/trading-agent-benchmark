import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-04-13')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
def build(kind,h):
 a=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(20,min_periods=15).std()
  if kind=='reversal': f=(-(c.pct_change(3))/(vol*np.sqrt(3)+1e-12)).shift(1)
  elif kind=='range_reversal': f=(-(c.pct_change(5))/( (d.high-d.low).rolling(20,min_periods=15).mean()/c +1e-12)).shift(1)
  elif kind=='accel': f=((c.pct_change(10)-c.pct_change(30)/3)/(vol+1e-12)).shift(1)
  fr=c.shift(-h)/c-1
  a.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':fr}))
 return pd.concat(a,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(q):
 z=[]; n=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); n.append(len(g))
 z=pd.Series(z)
 return len(z),np.mean(n),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
print('assets',len(D),[(s,len(x)) for s,x in D.items()])
for kind in ['reversal','range_reversal','accel']:
 q=build(kind,1); print('\n',kind,'rows',len(q),'dates',q.date.nunique(),'coverage',len(q)/(q.date.nunique()*15),'turn',q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True).diff().abs().mean().mean())
 for h in [1,5,10,20]: print(h,stats(build(kind,h)))
 for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print(a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
 q.to_csv('scripts/miner_2_20270414_'+kind+'_signal.csv',index=False)

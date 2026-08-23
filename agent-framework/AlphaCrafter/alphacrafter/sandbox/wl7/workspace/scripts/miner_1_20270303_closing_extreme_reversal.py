import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-02')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize()
    return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
# Close-location weighted short reversal: reverse lagged 3-session return, with
# larger weight after an impulsive close near the day's extreme. All inputs lagged.
rows=[]
for s,d in D.items():
 c=d.close; r=c.pct_change(); rv=r.rolling(20,min_periods=15).std()
 rng=(d.high-d.low).replace(0,np.nan); cl=((d.close-d.low)/rng).clip(0,1)
 impulse=(2*cl-1).abs().rolling(3,min_periods=2).mean()
 sig=((-c.pct_change(3)/(rv*np.sqrt(3)+1e-12))*(1+0.5*impulse)).shift(1)
 for h in [1,5,10,20]:
  rows.append(pd.DataFrame({'date':c.index,'asset':s,'signal':sig,'forward_return':c.shift(-h)/c-1,'h':h}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def calc(x):
 a=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1 and g.forward_return.nunique()>1:
   a.append(g.signal.corr(g.forward_return,method='spearman')); ns.append(len(g))
 z=pd.Series(a); return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
base=q[q.h==1]; print('assets',len(D),'dates',base.date.nunique(),'avg_n',base.groupby('date').size().mean(),'coverage',len(base)/(base.date.nunique()*len(D)))
for h in [1,5,10,20]: print('horizon',h,calc(q[q.h==h]))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',lo,hi,calc(base[(base.date.dt.year>=lo)&(base.date.dt.year<=hi)]))
p=base.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean().mean())
base.to_csv('scripts/miner_1_20270303_closing_extreme_reversal_signal.csv',index=False)

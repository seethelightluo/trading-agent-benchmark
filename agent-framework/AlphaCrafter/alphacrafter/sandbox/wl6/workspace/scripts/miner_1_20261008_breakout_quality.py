import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try:
   x=f(s,2600)
   if x is not None and len(x): return x
  except: pass
D={s:fetch(s) for s in U}; out=[]
for s,x in D.items():
 if x is None: continue
 x=x.copy(); x.date=pd.to_datetime(x.date); x=x.sort_values('date').drop_duplicates('date'); c=pd.to_numeric(x.close,errors='coerce'); h=pd.to_numeric(x.high,errors='coerce'); l=pd.to_numeric(x.low,errors='coerce')
 # breakout quality: distance from prior 60d high, penalized by recent range volatility; all lagged
 prior_high=h.rolling(60).max().shift(1); atr=(h-l).rolling(20).mean().shift(1)
 f=(c.shift(1)-prior_high)/atr
 fr=np.log(c.shift(-1)/c); out.append(pd.DataFrame({'date':x.date,'asset':s,'f':f,'fr':fr}))
a=pd.concat(out).dropna(); vals=[]
for d,g in a.groupby('date'):
 if len(g)>=8: vals.append(g.f.corr(g.fr,method='spearman'))
v=np.array(vals); ranks=a.assign(r=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='asset',values='r'); turn=ranks.diff().abs().mean(axis=1).mean()
print('dates',len(v),'assets/date',a.groupby('date').size().mean(),'coverage',len(a)/(15*a.date.nunique()),'daily_ic',np.nanmean(v),'icir',np.nanmean(v)/np.nanstd(v,ddof=1),'hit',np.mean(v>0),'turnover',turn)
for H in [5,10]:
 q=[]
 for s,x in D.items():
  if x is None: continue
  x=x.copy(); x.date=pd.to_datetime(x.date); c=pd.to_numeric(x.close,errors='coerce'); h=pd.to_numeric(x.high,errors='coerce'); l=pd.to_numeric(x.low,errors='coerce'); f=(c.shift(1)-h.rolling(60).max().shift(1))/(h-l).rolling(20).mean().shift(1); q.append(pd.DataFrame({'date':x.date,'f':f,'r':np.log(c.shift(-H)/c)}))
 q=pd.concat(q).dropna(); z=np.array([g.f.corr(g.r,method='spearman') for _,g in q.groupby('date') if len(g)>=8]); print('horizon',H,'dates',len(z),'ic',np.nanmean(z),'icir',np.nanmean(z)/np.nanstd(z,ddof=1))

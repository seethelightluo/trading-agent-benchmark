import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=3000)
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index()
D={s:load(s) for s in U};D={s:d for s,d in D.items() if d is not None}
# Defensive beta: negative rolling beta to equal-weight tradable benchmark return.
R=pd.concat({s:d.close.pct_change() for s,d in D.items()},axis=1).sort_index(); m=R.mean(axis=1)
for w in [40,60,120]:
 rows=[]
 for s in D:
  cov=R[s].rolling(w,min_periods=max(25,w//2)).cov(m); var=m.rolling(w,min_periods=max(25,w//2)).var()
  f=-cov/var
  z=pd.DataFrame({'f':f,'y':R[s].shift(-1)}).dropna();z['s']=s;rows.append(z.reset_index(names='date'))
 X=pd.concat(rows)
 a=[]; ns=[]
 for dt,g in X.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:a.append(g.f.corr(g.y));ns.append(len(g))
 a=np.asarray(a);print('beta',w,'dates',len(a),'avg',np.mean(ns),'coverage',len(X)/(len(R)*len(D)),'IC %.5f ICIR %.5f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
 for h in [5,10]:
  aa=[]
  for s in D:
   cov=R[s].rolling(w,min_periods=max(25,w//2)).cov(m);var=m.rolling(w,min_periods=max(25,w//2)).var();f=-cov/var
   z=pd.DataFrame({'f':f,'y':R[s].shift(-h).rolling(h).sum()}).dropna();z['date']=z.index;z['s']=s
   # collect later
  Y=[]
  for s in D:
   cov=R[s].rolling(w,min_periods=max(25,w//2)).cov(m);var=m.rolling(w,min_periods=max(25,w//2)).var();f=-cov/var
   Y.append(pd.DataFrame({'date':R.index,'f':f.values,'y':R[s].shift(-1).rolling(h).sum().values,'s':s}))
  Z=pd.concat(Y);aa=[]
  for dt,g in Z.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1:aa.append(g.f.corr(g.y))
  aa=np.asarray(aa);print(' h',h,'IC %.5f ICIR %.5f'%(np.nanmean(aa),np.nanmean(aa)/np.nanstd(aa,ddof=1)))

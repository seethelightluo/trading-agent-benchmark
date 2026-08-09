"""One candidate: continuous drawdown-recovery persistence (60/10).
Assets recovering over 10 days after a deeper 60-day drawdown may retain relative
strength, while the continuous drawdown term avoids sparse event gates.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index();R=P.pct_change();cut=P.index.max()
dd=P/P.rolling(60,min_periods=45).max()-1
# Standardize recovery by own recent variation; rank-neutralize drawdown each date.
vol=R.rolling(20,min_periods=15).std().replace(0,np.nan)
recovery=P.pct_change(10).div(vol*np.sqrt(10))
rel_dd=dd.sub(dd.median(axis=1),axis=0)
# Positive score: a recent normalized recovery proportional to prior relative damage.
F=(recovery*(-rel_dd)).replace([np.inf,-np.inf],np.nan).shift(1)
F=F.sub(F.median(axis=1),axis=0)
def metric(h,lo=None,hi=None,sign=1):
 x=(sign*F).loc[lo:hi];y=(P.shift(-h)/P-1).reindex(x.index);vs=[];ns=[]
 for t in x.index:
  z=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>2:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v):vs.append(v);ns.append(len(z))
 if not vs:return {'dates':0}
 v=np.array(vs);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR continuous_drawdown_recovery_persistence_60_10 cutoff',cut.date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for s,n in [(1,'persistence'),(-1,'inverse')]:
 print('ORIENTATION',n)
 for h in (1,5,10,20):print('H',h,metric(h,sign=s))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]:print('REGIME10',n,metric(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
print('LIBRARY_CORRELATION deferred unless performance and regime evidence pass.')

"""2031-02-06 current revalidation: inverse peer-relative directional consistency, 60 sessions."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def px(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date']).dt.normalize()
 return pd.Series(pd.to_numeric(d['close'],errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:px(a) for a in A}).sort_index(); R=P.pct_change()
F=(-np.sign(R.sub(R.median(axis=1),axis=0)).rolling(60,min_periods=45).mean()).shift(1)
F=F.sub(F.median(axis=1),axis=0)
def st(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); vals=[]; ns=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z);ns.append(len(q))
 if not vals:return {'ic_dates':0}
 v=np.array(vals);return {'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit_ratio':round(float((v>0).mean()),4),'ic_dates':len(v),'mean_instruments':round(float(np.mean(ns)),3),'minimum_instruments':min(ns)}
cut=P.index.max(); print('FACTOR inverse_peer_relative_directional_consistency_60');print('VALIDATED 2031-02-06 ENDPOINT',cut.date(),'UNIVERSE',len(A),'DATES',len(P));print('COVERAGE',round(float(F.notna().stack().mean()),6),'CELLS',int(F.notna().sum().sum()))
for h in (1,5,10,20):print('H',h,st(h))
for name,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent_180_calendar',str(cut-pd.Timedelta(days=180)),None)]:print('REGIME10',name,st(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6))

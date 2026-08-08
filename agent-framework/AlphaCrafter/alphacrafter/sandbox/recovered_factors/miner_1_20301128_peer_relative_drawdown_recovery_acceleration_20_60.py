"""One candidate: peer-relative drawdown recovery acceleration (20,60).
Measures rebound from the recent 20-session trough, scaled by the still-open
60-session drawdown. High values identify assets that are recovering rapidly
while remaining below their medium-term peak; the cross-sectional hypothesis is
that persistent recoveries continue over subsequent holding periods.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); cut=P.index.max()
low20=P.rolling(20,min_periods=15).min(); peak60=P.rolling(60,min_periods=45).max()
# Recovery is nonnegative; denominator makes signals comparable across assets
# and remains stable with an absolute 60d realized-return-scale floor.
scale=R.rolling(60,min_periods=45).std()*np.sqrt(60)
rebound=P/low20-1
open_dd=(peak60-P)/peak60
raw=rebound/(open_dd+scale*0.25+0.0025)
F=raw.sub(raw.median(axis=1),axis=0).shift(1)
def metric(h,lo=None,hi=None,sign=1):
 x=(sign*F).loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); z=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z); return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR peer_relative_drawdown_recovery_acceleration_20_60 cutoff',cut.date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for s,n in [(1,'recovery_acceleration'),(-1,'inverse_recovery_acceleration')]:
 print('ORIENTATION',n)
 for h in (1,5,10,20): print('H',h,metric(h,sign=s))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]: print('REGIME10',n,metric(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
print('LIBRARY_CORRELATION unavailable: admitted JSON files do not retain aligned daily signal panels.')

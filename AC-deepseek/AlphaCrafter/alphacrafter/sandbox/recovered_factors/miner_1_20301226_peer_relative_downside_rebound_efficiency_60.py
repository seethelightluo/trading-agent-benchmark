"""One candidate: peer-relative downside rebound efficiency (60 sessions).
For each asset, measure its mean next-session return following its own negative
sessions over a trailing 60-session window, then subtract the peer-median
value. This tests whether cross-asset-specific short-horizon downside recovery
propensity persists at a medium investment horizon. Inputs are lagged one day.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); cutoff=P.index.max()
# At t, the latest eligible event is t-1 and its rebound return is r_t;
# shift after rolling makes the full observation set strictly completed at signal time.
event=R.shift(1)<0
rebound=R.where(event)
raw=rebound.rolling(60,min_periods=12).mean()
F=raw.sub(raw.median(axis=1),axis=0).shift(1)
def metrics(h,lo=None,hi=None,sgn=1):
 x=(F*sgn).loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); vals=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);ns.append(len(q))
 if not vals:return {'dates':0}
 z=np.array(vals);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR peer_relative_downside_rebound_efficiency_60 cutoff',cutoff.date(),'assets',len(A),'price_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for sign,name in [(1,'continuation'),(-1,'inverse')]:
 print('ORIENTATION',name)
 for h in (1,5,10,20): print('H',h,metrics(h,sgn=sign))
print('REGIMES / 10-session best orientation selected only after full-sample output')
for name,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cutoff-pd.Timedelta(days=180)),None)]:
 print(name,'direct',metrics(10,lo,hi,1),'inverse',metrics(10,lo,hi,-1))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTION_SD',round(float(F.std(axis=1).mean()),6))
print('Novelty correlation deliberately not calculated unless a paper orientation and horizon clear both gates.')

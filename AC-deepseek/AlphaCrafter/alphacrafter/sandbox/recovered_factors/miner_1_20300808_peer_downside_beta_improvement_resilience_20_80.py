"""One candidate: peer downside-beta improvement resilience (20/80).
Test whether assets whose sensitivity to broad cross-asset down days has fallen
recently, while retaining non-negative peer-relative trend, outperform. The
change form aims to distinguish adaptive defensiveness from static low-beta.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); cut=P.index.max()
# Equal peer return is deliberately leave-one-out to prevent own-return mechanical beta.
B=pd.DataFrame(index=R.index,columns=A,dtype=float)
for a in A:
 peer=R.drop(columns=a).mean(axis=1); neg=peer.where(peer<0)
 # slope on negative peer days, short and long windows; require sufficient stress days.
 cov20=R[a].rolling(20,min_periods=12).cov(neg); var20=neg.rolling(20,min_periods=12).var()
 cov80=R[a].rolling(80,min_periods=42).cov(neg); var80=neg.rolling(80,min_periods=42).var()
 B[a]=(cov80/var80)-(cov20/var20) # positive: beta recently declined
# Require modest contemporaneous relative trend so a beta fall caused by persistent lagging is discounted.
 rel20=R.rolling(20,min_periods=16).sum().sub(R.rolling(20,min_periods=16).sum().median(axis=1),axis=0)
 trend=np.tanh(rel20/(R.rolling(20,min_periods=16).std()*np.sqrt(20)).replace(0,np.nan))
F=(B* (0.5+0.5*trend)).sub((B*(0.5+0.5*trend)).median(axis=1),axis=0).shift(1)
def metric(h,lo=None,hi=None,sign=1):
 x=(F*sign).loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); vals=[];ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):vals.append(v);ns.append(len(q))
 if not vals:return {'dates':0}
 v=np.array(vals);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR peer_downside_beta_improvement_resilience_20_80 cutoff',cut.date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for s,n in [(1,'beta_improvement'),(-1,'inverse')]:
 print('ORIENTATION',n)
 for h in (1,5,10,20):print('H',h,metric(h,sign=s))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]:print('REGIME10',n,metric(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(F.std(axis=1).mean()),6))
print('LIBRARY_CORRELATION deferred unless predictive and stability gates pass.')

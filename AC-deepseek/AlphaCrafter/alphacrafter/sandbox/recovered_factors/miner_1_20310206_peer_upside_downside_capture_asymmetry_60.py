"""One candidate: peer-upside/downside capture asymmetry, 60 sessions.
For each asset, calculate mean relative return (asset minus cross-asset median)
on broad peer-up days divided by the absolute mean relative return on peer-down
days. High values denote asymmetric peer-relative participation: leadership in
risk-on periods without comparable underperformance in risk-off periods. The
ratio is robustly cross-sectionally centered and lagged one completed session.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); cutoff=P.index.max()
peer=R.median(axis=1); rel=R.sub(peer,axis=0)
up=rel.where(peer.gt(0)); down=rel.where(peer.lt(0))
# Minimum observations in both conditional states protects early sparse windows.
u=up.rolling(60,min_periods=12).mean(); d=down.rolling(60,min_periods=12).mean()
raw=u/(d.abs()+1e-5)
# Winsorize only through same-date cross-sectional quantiles, then median center.
raw=raw.clip(raw.quantile(.05,axis=1),raw.quantile(.95,axis=1),axis=0)
F=raw.sub(raw.median(axis=1),axis=0).shift(1)
def met(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); vals=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);ns.append(len(q))
 if not vals:return {'dates':0}
 z=np.array(vals); return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR peer_upside_downside_capture_asymmetry_60 cutoff',cutoff.date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6),'up_freq',round(float(peer.gt(0).mean()),4),'down_freq',round(float(peer.lt(0).mean()),4))
for h in (1,5,10,20): print('H',h,met(h))
print('REGIMES horizon10')
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cutoff-pd.Timedelta(days=180)),None)]: print(n,met(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTION_SD',round(float(F.std(axis=1).mean()),6))

"""One candidate: copper-upside beta momentum, 60 sessions.
Positive beta to COPPER returns on COPPER-up days, cross-sectionally centered and lagged.
This is the sign inversion of the previously rejected resilience formulation.
"""
import runpy
import numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20300124_inverse_peer_relative_recovery_path_efficiency_60_library.py')
P,r,lib=z['P'],z['r'],z['lib']; A=list(P)
cs=lambda x:x.sub(x.median(axis=1),axis=0)
cop=r['COPPER']; event=cop.where(cop>0)
def beta(x,y,w=60):
 return x.rolling(w,min_periods=35).cov(y)/y.rolling(w,min_periods=35).var().replace(0,np.nan)
F=cs(pd.DataFrame({a:beta(r[a].where(cop>0),event) for a in A})).shift(1)
def met(h,lo=None,hi=None):
 vals=[]; ns=[]; y=P.shift(-h).div(P).sub(1)
 for t in F.loc[lo:hi].index:
  q=pd.concat([F.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v); ns.append(len(q))
 if not vals:return {'dates':0}
 v=np.array(vals); return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))}
print('FACTOR copper_upside_beta_momentum_60 cutoff',P.index.max().date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6),'copper_positive_days',int((cop>0).sum()))
for h in (1,5,10,20): print('H',h,met(h))
cut=P.index.max()
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]: print('REGIME10',n,met(10,lo,hi))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_SD',round(float(F.std(axis=1).mean()),6))
mx=-1;who=''; evidence=0; missing=[]
for n,g in lib.items():
 q=pd.concat([F.stack(),g.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho):
  evidence+=1
  if abs(rho)>mx: mx,who=abs(rho),n
 else: missing.append(n)
print('WHOLE_LIBRARY_RECONSTRUCTED',len(lib),'valid',evidence,'missing',missing,'MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),who)

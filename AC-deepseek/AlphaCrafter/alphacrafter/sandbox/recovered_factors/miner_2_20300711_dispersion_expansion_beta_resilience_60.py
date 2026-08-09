"""One idea: cross-asset dispersion-expansion beta resilience, 60 sessions."""
import runpy
import numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20300124_inverse_peer_relative_recovery_path_efficiency_60_library.py')
P,r,rel,lib=z['P'],z['r'],z['rel'],z['lib']; A=list(P)
# Daily cross-sectional dispersion is a macro state. During a rising, unusually
# high dispersion episode, score assets with lower beta to the dispersion shock.
disp=r.std(1); ds=disp.rolling(10,min_periods=7).std(); event=(disp>disp.rolling(60,min_periods=35).quantile(.75))&(disp>disp.shift(5))
shock=disp.diff().where(event)
def beta(x,y,w): return x.rolling(w,min_periods=15).cov(y)/y.rolling(w,min_periods=15).var()
F=pd.DataFrame({a:-beta(r[a].where(event),shock,60) for a in A})
F=F.sub(F.median(1),axis=0).shift(1)
def met(h,lo=None,hi=None):
 out=[]; ns=[]
 for t in F.loc[lo:hi].index:
  q=pd.concat([F.loc[t],(P.shift(-h)/P-1).loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):out.append(v);ns.append(len(q))
 if not out:return {'dates':0}
 x=np.array(out);return {'dates':len(x),'ic':round(x.mean(),6),'icir':round(x.mean()/x.std(ddof=1),6),'hit':round((x>0).mean(),4),'mean_n':round(np.mean(ns),2),'min_n':min(ns)}
print('CANDIDATE dispersion_expansion_beta_resilience_60 cutoff',P.index.max().date(),'assets',len(A),'calendar_dates',len(P))
print('CELLS',F.notna().sum().sum(),'/',F.size,'coverage',round(F.notna().stack().mean(),6),'event_dates',int(event.sum()))
for h in (1,5,10,20):print('H',h,met(h))
cut=P.index.max()
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]:print('REGIME10',n,met(10,lo,hi))
print('TURNOVER',round(F.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_SD',round(F.std(1).mean(),6))
mx=-1;who='';valid=0
for n,g in lib.items():
 q=pd.concat([F.stack(),g.stack()],axis=1).dropna();v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(v):valid+=1;print('LIBCORR',n,'cells',len(q),'rho',round(v,6));
 if np.isfinite(v) and abs(v)>mx:mx=abs(v);who=n
print('WHOLE_LIBRARY_RECONSTRUCTED',len(lib),'valid',valid,'MAX_ABS',round(mx,6),who)

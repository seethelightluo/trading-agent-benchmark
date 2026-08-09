"""One idea: library-orthogonal peer-upside resilience, 60 sessions.
Measures an asset's abnormal return when its other-14 peer basket rises, scaled by
own volatility, then removes contemporaneous linear exposures to the admitted
library.  A broad conditional resilience construct rather than an event filter.
"""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff,cs=z['P'],z['r'],z['S'],z['A'],z['cutoff'],z['cs']
peer=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A})
rel=r.sub(peer)
# Conditional peer-upside mean, risk normalized.  At least 15 observations
# avoids a thin conditional sample while retaining broad daily coverage.
raw=pd.DataFrame({a:rel[a].where(peer[a]>0).rolling(60,min_periods=15).mean()/r[a].rolling(20,min_periods=15).std() for a in A})
raw=cs(raw)
# Cross-sectional residual against the full operational library at each date.
# Use ridge-stabilized OLS and require substantial cross-sectional breadth.
cand=pd.DataFrame(np.nan,index=raw.index,columns=A)
for dt in raw.index:
    y=raw.loc[dt]
    X=pd.DataFrame({name:g.loc[dt] for name,g in S.items()})
    q=pd.concat([y.rename('y'),X],axis=1).dropna()
    if len(q)<8: continue
    # Drop constant/duplicate numerical columns; factor is defined as residual.
    xx=q.iloc[:,1:]; xx=xx.loc[:,xx.std()>1e-12]
    if xx.shape[1]:
        Z=np.column_stack([np.ones(len(q)),xx.to_numpy(float)])
        lam=1e-7*np.trace(Z.T@Z)/Z.shape[1]
        b=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@q.y.to_numpy(float))
        cand.loc[dt,q.index]=q.y.to_numpy(float)-Z@b
    else: cand.loc[dt,q.index]=q.y
cand=cs(cand).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); vals=[];breadth=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):vals.append(v);breadth.append(len(q))
 if not vals:return {'dates':0}
 vals=np.array(vals);return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR library_orthogonal_peer_upside_resilience_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20): print('H',h,st(h))
for name,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',name,st(10,p))
mx=-1;who='';ev=0
for name,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=name;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))

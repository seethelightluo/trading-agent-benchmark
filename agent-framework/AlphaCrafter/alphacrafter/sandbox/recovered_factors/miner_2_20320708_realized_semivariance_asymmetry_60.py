"""One idea: asset-specific realized upside/downside semivariance asymmetry (60d)."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_3_20310821_moderate_yield_spread_postshock_relative_persistence_60.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
# Ratio of mean squared positive to negative daily moves: persistent upside-vs-downside path asymmetry.
pos=r.clip(lower=0).pow(2).rolling(60,min_periods=40).mean()
neg=(-r.clip(upper=0)).pow(2).rolling(60,min_periods=40).mean()
raw=np.log((pos+1e-10)/(neg+1e-10))
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h).div(P).sub(1) for h in (1,5,10,20)}
def stat(h,lo=None,hi=None):
 x=cand.loc[lo:hi] if lo else cand; out=[]; br=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):out.append(v);br.append(len(q))
 if not out:return {'dates':0}
 v=np.asarray(out); return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'breadth':round(float(np.mean(br)),3),'min_breadth':int(min(br))}
rk=cand.rank(axis=1,pct=True)
print('FACTOR realized_semivariance_asymmetry_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(rk.diff().abs().stack().mean()),6))
for h in (1,5,10,20):print('H',h,stat(h))
for nm,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cutoff.date())),('recent180',str(cutoff-pd.Timedelta(days=180)),str(cutoff.date()))]: print('REGIME10',nm,stat(10,lo,hi))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
for h in (1,5,10,20):
 s=stat(h); print('ADMISSION',h,abs(s.get('ic',0))>=.007 and abs(s.get('icir',0))>=.084 and mx<.5)

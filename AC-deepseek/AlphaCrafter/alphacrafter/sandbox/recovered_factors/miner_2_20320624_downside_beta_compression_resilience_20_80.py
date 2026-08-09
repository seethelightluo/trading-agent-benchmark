"""One idea: downside-beta compression resilience, short-vs-long downside beta."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_3_20310821_moderate_yield_spread_postshock_relative_persistence_60.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
m=r.median(axis=1)
def dbeta(x,w):
 q=pd.concat([x.rename('x'),m.rename('m')],axis=1).where(lambda d:d.m<0)
 return q.x.rolling(w,min_periods=max(8,w//3)).cov(q.m)/q.m.rolling(w,min_periods=max(8,w//3)).var()
# A lower recent than structural downside beta identifies assets whose downside linkage is compressing.
short=pd.DataFrame({a:dbeta(r[a],20) for a in A}); long=pd.DataFrame({a:dbeta(r[a],80) for a in A})
cand=(long-short).sub((long-short).median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h).div(P).sub(1) for h in (1,5,10,20)}
def stat(h,lo=None,hi=None):
 x=cand.loc[lo:hi] if lo else cand; vals=[]; breadth=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):vals.append(v);breadth.append(len(q))
 if not vals:return {'dates':0}
 vals=np.asarray(vals);return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
rk=cand.rank(axis=1,pct=True)
print('FACTOR downside_beta_compression_resilience_20_80 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(rk.diff().abs().stack().mean()),6))
for h in (1,5,10,20):print('H',h,stat(h))
for nm,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cutoff.date())),('recent180',str(cutoff-pd.Timedelta(days=180)),str(cutoff.date()))]:print('REGIME10',nm,stat(10,lo,hi))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
s10,s20=stat(10),stat(20)
print('ADMISSION_10',abs(s10.get('ic',0))>=.007 and abs(s10.get('icir',0))>=.084 and mx<.5)
print('ADMISSION_20',abs(s20.get('ic',0))>=.007 and abs(s20.get('icir',0))>=.084 and mx<.5)

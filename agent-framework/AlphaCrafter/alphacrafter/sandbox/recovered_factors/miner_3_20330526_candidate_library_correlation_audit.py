"""Admission correlation audit for Miner_3 20330526 candidate against all locally available signal artifacts."""
import glob,pandas as pd,numpy as np
from scipy.stats import spearmanr
cand=pd.read_pickle('scripts/miner_3_20330526_high_dispersion_broad_drawdown_residual_pullback_reversal_5_20_60_signal.pkl')
rows=[]
for fn in glob.glob('scripts/*signal.pkl'):
 if '20330526_' in fn: continue
 try:
  x=pd.read_pickle(fn)
  if not isinstance(x,pd.DataFrame): continue
  a,b=cand.align(x,join='inner',axis=None) if False else (cand,x)
  # align both date and symbol dimensions
  a,b=a.align(b,join='inner',axis=0); a,b=a.align(b,join='inner',axis=1)
  q=pd.DataFrame({'a':a.stack(), 'b':b.stack()}).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1:
   rho=spearmanr(q.a,q.b).statistic
   rows.append((abs(rho),rho,len(q),fn))
 except Exception as e: print('SKIP',fn,type(e).__name__)
for z in sorted(rows,reverse=True): print('CORR abs=%.6f rho=%+.6f cells=%d file=%s'%z)
print('AVAILABLE_ARTIFACTS_SCREENED',len(rows),'OF',len(glob.glob('scripts/*signal.pkl'))-1)
PY
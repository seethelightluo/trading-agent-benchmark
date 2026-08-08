"""Independence audit: inverse residual rebound-efficiency candidate vs all active factors."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
f=pd.read_pickle('scripts/miner_1_20331013_inverse_residual_rebound_efficiency_20_60_signal.pkl')
active=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except: pass
arts=[]
for fid in active:
 h=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if h: arts.append((fid,max(h,key=os.path.getmtime)))
 else: print('MISSING_ACTIVE_ARTIFACT',fid)
print('AUDIT candidate_cells',int(f.notna().sum().sum()),'active_definitions',len(active),'artifacts_found',len(arts))
rows=[]
for fid,fn in arts:
 try:
  x=pd.read_pickle(fn); ds=f.index.intersection(x.index);cs=f.columns.intersection(x.columns); vals=[];ns=[]
  for d in ds:
   q=pd.concat([f.loc[d,cs].rename('a'),x.loc[d,cs].rename('b')],axis=1).dropna()
   if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1:
    z=spearmanr(q.a,q.b).statistic
    if np.isfinite(z):vals.append(abs(z));ns.append(len(q))
  if vals:rows.append((fid,max(vals),len(vals),np.mean(ns)))
  else:print('NO_OVERLAP_EVIDENCE',fid)
 except Exception as e: print('ARTIFACT_ERROR',fid,type(e).__name__)
for a,b,c,d in sorted(rows,key=lambda z:-z[1]):print('COMPARE',a,'max_abs_rho=%.6f dates=%d meanN=%.2f'%(b,c,d))
best=max(rows,key=lambda z:z[1]);print('RESULT max_abs_library_correlation=%.6f factor=%s compared=%d required=%d'%(best[1],best[0],len(rows),len(arts)))

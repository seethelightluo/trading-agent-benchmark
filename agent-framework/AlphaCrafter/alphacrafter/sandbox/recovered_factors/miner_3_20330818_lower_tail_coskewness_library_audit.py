"""Exact independence audit for the previously validated lower-tail co-skewness candidate.
Only active admitted factor definitions are in scope, rather than every experimental artifact.
"""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
candidate='scripts/miner_3_20330804_lower_tail_coskewness_contraction_20_60_signal.pkl'
f=pd.read_pickle(candidate)
active=[]
for fn in glob.glob('factors/*.json'):
    try:
        j=json.load(open(fn))
        if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
    except Exception: pass
# An admitted definition may have a date prefix different from its artifact; match suffix factor_id exactly.
artifacts=[]
for fid in active:
    hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
    if not hits:
        print('MISSING_ACTIVE_ARTIFACT',fid)
    else:
        artifacts.append((fid,max(hits,key=os.path.getmtime)))
print('AUDIT candidate_cells',int(f.notna().sum().sum()),'active_definitions',len(active),'artifacts_found',len(artifacts))
maxrho=-1; maxname=None; compared=0; evidence=[]
for fid,fn in artifacts:
    try:
        x=pd.read_pickle(fn); ds=f.index.intersection(x.index); cs=f.columns.intersection(x.columns)
        vals=[]; nobs=[]
        for d in ds:
            q=pd.concat([f.loc[d,cs].rename('candidate'),x.loc[d,cs].rename('library')],axis=1).dropna()
            if len(q)>=8 and q.candidate.nunique()>1 and q.library.nunique()>1:
                z=spearmanr(q.candidate,q.library).statistic
                if np.isfinite(z): vals.append(abs(float(z)));nobs.append(len(q))
        if vals:
            z=max(vals); compared+=1
            evidence.append((fid,z,len(vals),float(np.mean(nobs))))
            if z>maxrho: maxrho,maxname=z,fid
        else: print('NO_OVERLAP_EVIDENCE',fid)
    except Exception as e: print('ARTIFACT_ERROR',fid,type(e).__name__)
for row in sorted(evidence,key=lambda z:-z[1]): print('COMPARE',row[0],'max_abs_rho=%.6f ic_dates=%d meanN=%.2f'%(row[1],row[2],row[3]))
print('RESULT max_abs_library_correlation=%.6f factor=%s compared=%d required_artifacts=%d'%(maxrho,maxname,compared,len(artifacts)))

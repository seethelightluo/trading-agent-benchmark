"""Full novelty audit for the validated inverse directional-consistency candidate.
Uses source scripts corresponding to every admitted JSON factor, executes them
against the same visible cursor, and extracts their matrix signal (f/F). This is
an audit only; failure to reconstruct any admitted signal means no admission.
"""
import glob, json, os, io, contextlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
# Candidate prepared exactly as the validation script.
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change()
F=(-np.sign(R.sub(R.median(axis=1),axis=0)).rolling(60,min_periods=45).mean()).shift(1)
F=F.sub(F.median(axis=1),axis=0)

def source_for(fid):
 hits=[]
 for p in glob.glob('scripts/*.py'):
  try:
   if fid in open(p,encoding='utf-8').read():hits.append(p)
  except:pass
 return sorted(hits,key=lambda x:os.path.getmtime(x),reverse=True)
mx=0.; who=''; evidence=0; failed=[]; audited=0
for jf in glob.glob('factors/*.json'):
 d=json.load(open(jf)); fid=d['factor_id']; hits=source_for(fid)
 if not hits:
  failed.append(fid+':no_source');continue
 try:
  with contextlib.redirect_stdout(io.StringIO()): ns=__import__('runpy').run_path(hits[0])
  g=ns.get('f',ns.get('F'))
  if not isinstance(g,pd.DataFrame):raise ValueError('no_dataframe_f_or_F')
  q=pd.concat([F.stack().rename('candidate'),g.stack().rename('library')],axis=1).dropna()
  if len(q)<3:raise ValueError('insufficient_overlap')
  rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if not np.isfinite(rho):raise ValueError('nonfinite_rho')
  audited+=1
  print('LIBRARY',fid,'SOURCE',os.path.basename(hits[0]),'CELLS',len(q),'RHO',round(float(rho),6))
  if abs(rho)>mx:mx=float(abs(rho));who=fid;evidence=len(q)
 except Exception as e:
  failed.append(fid+':'+str(e)[:100])
print('AUDITED',audited,'OF',len(glob.glob('factors/*.json')))
print('FAILED',failed)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',who,'EVIDENCE_CELLS',evidence)

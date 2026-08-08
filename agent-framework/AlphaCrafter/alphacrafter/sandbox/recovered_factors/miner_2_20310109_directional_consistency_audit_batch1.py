"""Incremental novelty audit, batch 1 of admitted library for inverse peer-relative directional consistency 60.
Writes durable correlation evidence so the full mandatory audit can span runtime-limited turns.
"""
import glob,json,os,io,contextlib,runpy
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim import utils
assets=utils.get_account_dict()['watch_list']
orig_stock,orig_index=utils.get_stock_daily_data,utils.get_index_daily_data
sc,ic={},{}
def stock(s,n=5000,*a,**k):
    if s not in sc: sc[s]=orig_stock(s,5000,*a,**k)
    return sc[s].tail(n).copy()
def index(s,n=5000,*a,**k):
    if s not in ic: ic[s]=orig_index(s,5000,*a,**k)
    return ic[s].tail(n).copy()
utils.get_stock_daily_data=stock;utils.get_index_daily_data=index
def cl(s):
 d=stock(s); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
p=pd.DataFrame({s:cl(s) for s in assets}).sort_index(); r=p.pct_change()
f=(-np.sign(r.sub(r.median(axis=1),axis=0)).rolling(60,min_periods=45).mean()).shift(1)
f=f.sub(f.median(axis=1),axis=0)
allf=[]
for x in glob.glob('factors/*.json'):
 try:
  d=json.load(open(x)); allf.append((d['factor_id'],x))
 except: pass
# current JSON files only; no backups. deterministic four batches
allf=sorted(allf); batch=allf[:8]
cache='scripts/miner_2_20310109_directional_consistency_audit_cache.json'
try: out=json.load(open(cache))
except: out={'candidate':'inverse_peer_relative_directional_consistency_60','endpoint':str(p.index.max().date()),'records':{},'failures':{}}
for fid,_ in batch:
 hits=[]
 for sp in glob.glob('scripts/*.py'):
  try:
   if fid in open(sp,encoding='utf8').read(): hits.append(sp)
  except: pass
 if not hits: out['failures'][fid]='no source'; print('FAIL',fid,'no_source'); continue
 src=sorted(hits,key=os.path.getmtime,reverse=True)[0]
 try:
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): ns=runpy.run_path(src)
  g=ns.get('f',ns.get('F'))
  if not isinstance(g,pd.DataFrame): raise ValueError('missing f/F DataFrame')
  q=pd.concat([f.stack().rename('candidate'),g.stack().rename('library')],axis=1).dropna()
  rho=spearmanr(q.candidate,q.library).statistic
  if not np.isfinite(rho): raise ValueError('nonfinite rho')
  out['records'][fid]={'rho':float(rho),'cells':len(q),'source':os.path.basename(src)}
  print('AUDIT',fid,'rho',round(float(rho),6),'cells',len(q))
 except Exception as e: out['failures'][fid]=str(e)[:180]; print('FAIL',fid,str(e)[:120])
json.dump(out,open(cache,'w'),indent=2)
print('BATCH_DONE records',len(out['records']),'failures',len(out['failures']),'required',len(allf),'cache',cache)

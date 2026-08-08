"""Cached, complete library novelty audit for inverse peer-relative directional consistency 60.
Caches visible OHLCV/index history once, then reconstructs every admitted factor from
its source script; failure for any factor explicitly fails admission.
"""
import glob,json,os,io,contextlib,runpy,sys
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim import utils
A=utils.get_account_dict()['watch_list']
# Cache all possibly requested tradable and observation symbols at the same cursor.
orig_stock,orig_index=utils.get_stock_daily_data,utils.get_index_daily_data
stock_cache={}; index_cache={}
def cstock(symbol,n=5000,*args,**kwargs):
 if symbol not in stock_cache: stock_cache[symbol]=orig_stock(symbol,5000,*args,**kwargs)
 return stock_cache[symbol].tail(n).copy()
def cindex(symbol,n=5000,*args,**kwargs):
 if symbol not in index_cache: index_cache[symbol]=orig_index(symbol,5000,*args,**kwargs)
 return index_cache[symbol].tail(n).copy()
utils.get_stock_daily_data=cstock;utils.get_index_daily_data=cindex
# candidate

def close(a):
 d=cstock(a,5000); d=d.copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:close(a) for a in A}).sort_index();R=P.pct_change()
F=(-np.sign(R.sub(R.median(axis=1),axis=0)).rolling(60,min_periods=45).mean()).shift(1)
F=F.sub(F.median(axis=1),axis=0)
def source_for(fid):
 hits=[]
 for p in glob.glob('scripts/*.py'):
  try:
   if fid in open(p,encoding='utf-8').read(): hits.append(p)
  except OSError: pass
 return sorted(hits,key=os.path.getmtime,reverse=True)
records=[]; failures=[]
for jf in glob.glob('factors/*.json'):
 d=json.load(open(jf)); fid=d['factor_id']; hits=source_for(fid)
 if not hits: failures.append(fid+':no_source');continue
 try:
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): ns=runpy.run_path(hits[0])
  g=ns.get('f',ns.get('F'))
  if not isinstance(g,pd.DataFrame): raise ValueError('no signal f/F')
  q=pd.concat([F.stack().rename('x'),g.stack().rename('y')],axis=1).dropna()
  rho=spearmanr(q.x,q.y).statistic if len(q)>=3 else np.nan
  if not np.isfinite(rho): raise ValueError('invalid correlation')
  records.append((fid,float(rho),len(q),os.path.basename(hits[0])))
  print('AUDIT',fid,'rho',round(float(rho),6),'cells',len(q))
 except Exception as e: failures.append(fid+':'+str(e)[:120])
print('AUDITED',len(records),'REQUIRED',len(glob.glob('factors/*.json')))
print('FAILURES',failures)
if records:
 z=max(records,key=lambda x:abs(x[1]));print('MAX',round(abs(z[1]),6),z[0],'CELLS',z[2])
print('PASS_COMPLETE',len(failures)==0 and len(records)==len(glob.glob('factors/*.json')))

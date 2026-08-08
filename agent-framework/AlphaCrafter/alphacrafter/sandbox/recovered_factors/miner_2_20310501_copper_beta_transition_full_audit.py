"""Full admitted-library novelty audit: copper beta transition 20/80.
Uses current visible cursor, source scripts for all admitted JSON factors, and fails on any reconstruction failure.
"""
import glob,json,os,io,contextlib,runpy
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim import utils
A=utils.get_account_dict()['watch_list']
orig_stock,orig_index=utils.get_stock_daily_data,utils.get_index_daily_data
stock_cache={}; index_cache={}
def cstock(symbol,n=5000,*args,**kwargs):
 if symbol not in stock_cache: stock_cache[symbol]=orig_stock(symbol,5000,*args,**kwargs)
 return stock_cache[symbol].tail(n).copy()
def cindex(symbol,n=5000,*args,**kwargs):
 if symbol not in index_cache: index_cache[symbol]=orig_index(symbol,5000,*args,**kwargs)
 return index_cache[symbol].tail(n).copy()
utils.get_stock_daily_data=cstock; utils.get_index_daily_data=cindex
def close(a):
 d=cstock(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:close(a) for a in A}).sort_index(); R=P.pct_change(); cr=R['COPPER']
def beta(x,y,w,mn): return x.rolling(w,min_periods=mn).cov(y)/y.rolling(w,min_periods=mn).var()
F=pd.DataFrame({a:beta(R[a],cr,20,15)-beta(R[a],cr,80,55) for a in A})
F=F.sub(F.median(axis=1),axis=0).shift(1)
def source_for(fid):
 hits=[]
 for p in glob.glob('scripts/*.py'):
  try:
   if fid in open(p,encoding='utf-8').read(): hits.append(p)
  except OSError: pass
 return sorted(hits,key=os.path.getmtime,reverse=True)
records=[];fail=[]; js=glob.glob('factors/*.json')
for jf in js:
 d=json.load(open(jf)); fid=d['factor_id']; hits=source_for(fid)
 if not hits: fail.append(fid+':no source'); continue
 try:
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): ns=runpy.run_path(hits[0])
  g=ns.get('f',ns.get('F'))
  if not isinstance(g,pd.DataFrame): raise ValueError('no DataFrame f/F')
  q=pd.concat([F.stack().rename('candidate'),g.stack().rename('library')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)<3: raise ValueError('fewer than 3 aligned cells')
  rho=spearmanr(q.candidate,q.library).statistic
  if not np.isfinite(rho): raise ValueError('nonfinite rho')
  records.append((fid,float(rho),len(q),os.path.basename(hits[0])))
  print('AUDIT',fid,'rho',round(float(rho),6),'cells',len(q),'source',os.path.basename(hits[0]))
 except Exception as e: fail.append(fid+':'+str(e)[:180])
print('AUDITED',len(records),'REQUIRED',len(js));print('FAILURES',fail)
if records:
 z=max(records,key=lambda x:abs(x[1]));print('MAX_ABS_LIBRARY_CORRELATION',round(abs(z[1]),6),'FACTOR',z[0],'EVIDENCE_CELLS',z[2])
print('PASS_COMPLETE',len(fail)==0 and len(records)==len(js))

"""Novelty audit for inverse_drawdown_recovery_exhaustion_60_10 against all admitted JSON factors."""
import glob,json,os,runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
# Candidate built identically to its validation script, no look-ahead.
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list'];C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C).sort_index(); dd=P.shift(10)/P.rolling(60,min_periods=45).max().shift(10)-1
cand=(-(P/P.shift(10)-1)*(-dd)).shift(1);cand=cand.sub(cand.median(axis=1),axis=0)
results=[]; skipped=[]
for jf in glob.glob('factors/*.json'):
 try:
  meta=json.load(open(jf))
  if meta.get('validation',{}).get('status')!='EFFECTIVE':continue
  stem=os.path.basename(jf)[:-5]; sf='scripts/'+stem+'.py'
  if not os.path.exists(sf):skipped.append((meta['factor_id'],'no source'));continue
  env=runpy.run_path(sf)
  sig=env.get('f',env.get('factor',None))
  if not isinstance(sig,pd.DataFrame):skipped.append((meta['factor_id'],'no dataframe f'));continue
  x,y=cand.align(sig,join='inner');q=pd.concat([x.stack(),y.stack()],axis=1).dropna()
  if len(q)<8:skipped.append((meta['factor_id'],'insufficient overlap'));continue
  rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  results.append((abs(rho),rho,meta['factor_id'],len(q)))
 except Exception as e:skipped.append((os.path.basename(jf),type(e).__name__))
results.sort(reverse=True)
print('EFFECTIVE_JSON',len(results)+len(skipped),'COMPARED',len(results),'SKIPPED',skipped)
for r in results:print('CORR',round(r[0],6),round(r[1],6),r[2],r[3])
print('MAX',results[0] if results else None)

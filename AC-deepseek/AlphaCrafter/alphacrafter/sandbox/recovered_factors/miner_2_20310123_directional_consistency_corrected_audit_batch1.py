"""Corrected novelty audit batch: use each factor's dedicated construction script, never a script that merely mentions the id."""
import json, os, io, contextlib, runpy
import numpy as np, pandas as pd
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
def close(s):
 d=stock(s); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
p=pd.DataFrame({s:close(s) for s in assets}).sort_index(); r=p.pct_change()
f=(-np.sign(r.sub(r.median(axis=1),axis=0)).rolling(60,min_periods=45).mean()).shift(1)
f=f.sub(f.median(axis=1),axis=0)
targets={
'commonality_shock_peer_relative_reversal_60':'scripts/miner_1_20280629_commonality_shock_peer_relative_reversal_60.py',
'delayed_post_stress_relative_rebound_reversal_60':'scripts/miner_1_20280518_delayed_post_stress_relative_rebound_reversal_60.py',
'dispersion_shock_peer_reversal_20':'scripts/miner_1_20280713_dispersion_shock_peer_reversal_20.py',
'downside_correlation_regime_spread_20_80':'scripts/miner_3_20280323_downside_correlation_regime_spread_20_80.py',
'dxy_directional_return_asymmetry_60':'scripts/miner_1_20271118_dxy_directional_return_asymmetry_60.py',
'gradual_volatility_contraction_gated_trend_20':'scripts/miner_1_20270701_gradual_volatility_contraction_gated_trend_20.py',
'inverse_lower_tail_persistence_40_60':'scripts/miner_3_20270826_inverse_lower_tail_persistence_40_60.py',
'inverse_peer_relative_serial_dependence_20':'scripts/miner_1_20300822_serial_dependence_library_audit.py'}
cache='scripts/miner_2_20310109_directional_consistency_audit_cache_corrected.json'
try: out=json.load(open(cache))
except: out={'candidate':'inverse_peer_relative_directional_consistency_60','endpoint':str(p.index.max().date()),'records':{},'failures':{}}
for fid,src in targets.items():
 try:
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): ns=runpy.run_path(src)
  g=ns.get('f',ns.get('F'))
  if not isinstance(g,pd.DataFrame): raise ValueError('missing f/F DataFrame')
  q=pd.concat([f.stack().rename('candidate'),g.stack().rename('library')],axis=1).dropna()
  rho=spearmanr(q.candidate,q.library).statistic
  if not np.isfinite(rho): raise ValueError('nonfinite rho')
  out['records'][fid]={'rho':float(rho),'cells':len(q),'source':os.path.basename(src)}
  print('AUDIT',fid,'rho',round(float(rho),6),'cells',len(q))
 except Exception as e:
  out['failures'][fid]=str(e)[:240]; print('FAIL',fid,str(e)[:160])
json.dump(out,open(cache,'w'),indent=2)
print('DONE',len(out['records']),'records',len(out['failures']),'failures')

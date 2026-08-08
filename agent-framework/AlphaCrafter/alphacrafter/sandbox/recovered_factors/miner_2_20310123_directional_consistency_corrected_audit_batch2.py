"""Novelty audit corrected batch 2: dedicated factor scripts and factor-specific DataFrame variables."""
import json,runpy,io,contextlib
import pandas as pd,numpy as np
from scipy.stats import spearmanr
from alphacrafter.sim import utils
A=utils.get_account_dict()['watch_list']; C={}
for a in A:
 d=utils.get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date).dt.normalize();C[a]=pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame(C).sort_index();r=P.pct_change();f=(-np.sign(r.sub(r.median(axis=1),axis=0)).rolling(60,min_periods=45).mean()).shift(1);f=f.sub(f.median(axis=1),axis=0)
items={
'miner_1_ravmom_20obs':'scripts/miner_1_20260716_ravmom_20obs.py','miner_1_volnorm_reversal_5obs':'scripts/miner_1_20260716_volnorm_reversal_5obs.py','vix_regime_conditioned_risk_adjusted_trend_20':'scripts/miner_1_20261203_vix_regime_conditioned_risk_adjusted_trend_20.py','miner_1_downside_cross_asset_beta_resilience_40':'scripts/miner_1_20270128_downside_cross_asset_beta_resilience_40.py','miner_1_inverse_idiosyncratic_volatility_20':'scripts/miner_1_20270225_inverse_idiosyncratic_volatility_20.py','stable_liquidity_participation_20':'scripts/miner_1_20270311_stable_liquidity_participation_20.py','return_skewness_60':'scripts/miner_1_20270422_return_skewness_60.py','post_stress_relative_rebound_reversal_60':'scripts/miner_1_20280504_post_stress_relative_rebound_reversal_60.py','inverse_volume_weighted_peer_tail_asymmetry_60':'scripts/miner_1_20280921_inverse_volume_weighted_peer_tail_asymmetry_60.py','yield_shock_beta_resilience_60':'scripts/miner_1_20291018_yield_shock_beta_resilience_60.py'}
path='scripts/miner_2_20310109_directional_consistency_audit_cache_corrected.json';out=json.load(open(path))
for fid,src in items.items():
 try:
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):ns=runpy.run_path(src)
  g=ns.get('f',ns.get('F',ns.get('factor',ns.get('signal'))))
  if not isinstance(g,pd.DataFrame): raise ValueError('no conventional factor DataFrame; keys='+','.join(ns.keys()))
  q=pd.concat([f.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  out['records'][fid]={'rho':float(rho),'cells':len(q),'source':src.split('/')[-1]};print('AUDIT',fid,round(rho,6),len(q))
 except Exception as e:out['failures'][fid]=str(e)[:180];print('FAIL',fid,str(e)[:100])
json.dump(out,open(path,'w'),indent=2);print('TOTAL',len(out['records']),'FAIL',len(out['failures']))

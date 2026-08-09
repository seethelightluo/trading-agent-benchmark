# miner_1: one-factor exploration -- cross-asset range-location acceleration
# High score identifies assets whose 5d return is improving versus 20d return while
# closing in the upper part of their own 20d range; a simple path-confirmed continuation signal.
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
A=get_account_dict()['watch_list']; series=[]
for a in A:
 d=get_stock_daily_data(a,5000)[['date','close']].copy(); d.date=pd.to_datetime(d.date)
 series.append(d.set_index('date').close.rename(a))
P=pd.concat(series,axis=1).sort_index(); END=P.index.max(); R=P.pct_change()
# acceleration normalized by 20d realized volatility, then multiplied by centered range position
acc=(R.rolling(5,min_periods=5).sum()-R.rolling(20,min_periods=20).sum())/(R.rolling(20,min_periods=15).std()+1e-12)
loc=(P-P.rolling(20,min_periods=20).min())/(P.rolling(20,min_periods=20).max()-P.rolling(20,min_periods=20).min()+1e-12)-.5
F=(acc*loc).rank(axis=1,pct=True).where(lambda x:x.count(axis=1)>=8)
def met(X,h):
 fw=np.log(P.shift(-h)/P); vals=[]; ns=[]
 for d in X.index:
  q=pd.concat([X.loc[d],fw.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v); ns.append(len(q))
 z=np.array(vals); sd=z.std(ddof=1)
 return {'daily_paper_ic':float(z.mean()),'daily_paper_icir':float(z.mean()/(sd+1e-12)),'ic_hit_ratio':float((z>0).mean()),'ic_dates':len(z),'mean_valid_instruments':float(np.mean(ns)),'ic_standard_error':float(sd/np.sqrt(len(z)))}
print('FACTOR cross_asset_range_location_acceleration_5v20obs')
print('VALIDATION_DATE',END.date(),'PERIOD',F.index.min().date(),END.date(),'ASSETS',len(A))
print('COVERAGE',float(F.notna().mean().mean()),'SIGNAL_DATES',int(F.notna().any(axis=1).sum()),'MEAN_NAMES',float(F.count(axis=1).mean()))
for h in [1,5,10,20,40]: print('HORIZON',h,json.dumps(met(F,h),sort_keys=True))
for n,l,r in [('2024_2026','2024-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2033','2031-01-01','2033-12-31'),('2034_current','2034-01-01',str(END.date()))]: print('REGIME',n,json.dumps(met(F.loc[l:r],20),sort_keys=True))
st=float(F.corrwith(F.shift(),axis=1,method='spearman').mean()); print('RANK_STABILITY_1D',st,'TURNOVER_PROXY',1-st)
out='scripts/miner_1_20350816_cross_asset_range_location_acceleration_5v20obs_signal.pkl';F.to_pickle(out)
# Required evidence audit: every currently effective library factor needs a usable aligned artifact.
alias={'miner_1_20260716_volnorm_reversal_5obs.json':'scripts/miner_1_20260716_volnorm_reversal5_signal.pkl','miner_1_20311211_state_gated_volatility_expansion_10v60obs.json':'scripts/miner_1_20320304_state_gated_volatility_expansion_10v60obs_signal.pkl','miner_2_20280127_standardized_jump_asymmetry_20v40obs.json':'scripts/miner_2_20280113_standardized_jump_asymmetry_20v40obs_signal.pkl'}
active=[]
for p in glob.glob('factors/*.json'):
 try:
  if json.load(open(p)).get('validation',{}).get('status')=='EFFECTIVE': active.append(os.path.basename(p))
 except: pass
base=F.stack().rename('x'); ev=[]; missing=[]
for f in active:
 paths=[alias[f]] if f in alias else glob.glob('scripts/*'+f[:-5]+'*_signal.pkl')
 if not paths: missing.append(f); continue
 try:
  y=pd.read_pickle(sorted(paths)[-1]); y=y.stack() if isinstance(y,pd.DataFrame) else y
  q=pd.concat([base,y.rename('y')],axis=1).dropna(); rho=abs(spearmanr(q.x,q.y).statistic)
  if len(q)<8 or not np.isfinite(rho): raise ValueError()
  ev.append((f,float(rho),len(q)))
 except: missing.append(f)
print('LIBRARY_AUDIT',json.dumps({'active':len(active),'evidence':len(ev),'missing':missing,'max_abs_library_correlation':max([z[1] for z in ev]) if ev else None,'most_correlated':max(ev,key=lambda z:z[1])[0] if ev else None,'complete':len(missing)==0},sort_keys=True))
print('TOP_CORRELATIONS',sorted(ev,key=lambda z:-z[1])[:10])

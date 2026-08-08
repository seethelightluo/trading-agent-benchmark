# miner_1: one-factor exploration -- yield-shock conditional inverse beta
# During a large 5d move in the tradable US10Y yield series, prefer assets with low
# trailing sensitivity to US10Y daily changes; otherwise retain a cross-sectionally
# neutral signal. This is an interpretable macro-risk-resilience characteristic.
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; xs=[]
for a in A:
 d=get_stock_daily_data(a,5000)[['date','close']].copy();d.date=pd.to_datetime(d.date)
 xs.append(d.set_index('date').close.rename(a))
P=pd.concat(xs,axis=1).sort_index(); END=P.index.max(); R=P.pct_change()
y=R['US10Y']; cov=R.rolling(40,min_periods=30).cov(y); beta=cov.div(y.rolling(40,min_periods=30).var()+1e-12,axis=0)
# Shock condition is known at date t only; 5d absolute move exceeds its 60d median.
shock=y.rolling(5,min_periods=5).sum().abs()>y.rolling(60,min_periods=45).apply(np.median,raw=True)
raw=(-beta).where(shock,0.0)
F=raw.rank(axis=1,pct=True).where(lambda x:x.count(axis=1)>=8)
def met(X,h):
 fw=np.log(P.shift(-h)/P); z=[];ns=[]
 for d in X.index:
  q=pd.concat([X.loc[d],fw.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.array(z);sd=z.std(ddof=1)
 return {'daily_paper_ic':float(z.mean()),'daily_paper_icir':float(z.mean()/(sd+1e-12)),'ic_hit_ratio':float((z>0).mean()),'ic_dates':len(z),'mean_valid_instruments':float(np.mean(ns)) if ns else 0.0,'ic_standard_error':float(sd/np.sqrt(len(z))) if len(z) else None}
print('FACTOR yield_shock_conditional_inverse_us10y_beta_5v40x60obs')
print('VALIDATION_DATE',END.date(),'PERIOD',F.index.min().date(),END.date(),'ASSETS',len(A),'SHOCK_RATE',float(shock.mean()))
print('COVERAGE',float(F.notna().mean().mean()),'SIGNAL_DATES',int(F.notna().any(axis=1).sum()),'MEAN_NAMES',float(F.count(axis=1).mean()))
for h in [1,5,10,20,40]:print('HORIZON',h,json.dumps(met(F,h),sort_keys=True))
for n,l,r in [('2024_2026','2024-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2033','2031-01-01','2033-12-31'),('2034_current','2034-01-01',str(END.date()))]:print('REGIME',n,json.dumps(met(F.loc[l:r],20),sort_keys=True))
st=float(F.corrwith(F.shift(),axis=1,method='spearman').mean());print('RANK_STABILITY_1D',st,'TURNOVER_PROXY',1-st)
out='scripts/miner_1_20350830_yield_shock_conditional_inverse_us10y_beta_5v40x60obs_signal.pkl';F.to_pickle(out)
alias={'miner_1_20260716_volnorm_reversal_5obs.json':'scripts/miner_1_20260716_volnorm_reversal5_signal.pkl','miner_1_20311211_state_gated_volatility_expansion_10v60obs.json':'scripts/miner_1_20320304_state_gated_volatility_expansion_10v60obs_signal.pkl','miner_2_20280127_standardized_jump_asymmetry_20v40obs.json':'scripts/miner_2_20280113_standardized_jump_asymmetry_20v40obs_signal.pkl'}
active=[]
for p in glob.glob('factors/*.json'):
 try:
  if json.load(open(p)).get('validation',{}).get('status')=='EFFECTIVE':active.append(os.path.basename(p))
 except:pass
base=F.stack().rename('x');ev=[];missing=[]
for f in active:
 paths=[alias[f]] if f in alias else glob.glob('scripts/*'+f[:-5]+'*_signal.pkl')
 if not paths:missing.append(f);continue
 try:
  y0=pd.read_pickle(sorted(paths)[-1]);y0=y0.stack() if isinstance(y0,pd.DataFrame) else y0
  q=pd.concat([base,y0.rename('y')],axis=1).dropna();rho=abs(spearmanr(q.x,q.y).statistic)
  if len(q)<8 or not np.isfinite(rho):raise ValueError()
  ev.append((f,float(rho),len(q)))
 except:missing.append(f)
print('LIBRARY_AUDIT',json.dumps({'active':len(active),'evidence':len(ev),'missing':missing,'max_abs_library_correlation':max([v[1] for v in ev]) if ev else None,'most_correlated':max(ev,key=lambda v:v[1])[0] if ev else None,'complete':len(missing)==0},sort_keys=True));print('TOP_CORRELATIONS',sorted(ev,key=lambda v:-v[1])[:10])

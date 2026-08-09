"""Miner_2: copper-regime-conditioned inflation sensitivity, cutoff 2033-03-16.
One interpretable idea: assets with positive trailing copper-return sensitivity should
outperform cross-sectionally when copper has a persistent positive impulse (and vice versa).
COPPER remains a tradable constituent; no observation-only series is traded here.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2033-03-16')
FID='miner_2_copper_regime_conditioned_inflation_sensitivity_10x60obs'
def close(path):
 d=pd.read_csv(path,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d['close'].astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index().loc[:END]
r=np.log(C).diff(); cr=r['COPPER']
# Stable 60-session beta to daily copper moves, scaled by signed 10-day copper impulse.
# This produces a cross-sectional conditional inflation/risk-cycle exposure.
beta=r.rolling(60,min_periods=45).cov(cr).div(cr.rolling(60,min_periods=45).var(),axis=0)
impulse=cr.rolling(10,min_periods=10).mean().div(cr.rolling(60,min_periods=45).std())
F=beta.mul(impulse,axis=0)
def evaluate(h):
 y=C.shift(-h).div(C).sub(1); vals=[]; widths=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): vals.append((d,float(q)));widths.append(len(z))
 ic=pd.Series(dict(vals),dtype=float); sd=ic.std(ddof=1)
 return ic, {'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd) if sd else None,'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))) if len(ic) else None,'ic_dates':int(len(ic)),'mean_valid_instruments':float(np.mean(widths)) if widths else 0.0}
M={}
for h in (1,5,10,20):
 _,M[h]=evaluate(h); print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic,_=evaluate(10)
for lab,yrs in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('recent_2032_2033',[2032,2033])]:
 x=ic[ic.index.year.isin(yrs)]; sd=x.std(ddof=1)
 print('REGIME_10D',lab,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/sd) if len(x)>1 and sd else None,'hit',float((x>0).mean()) if len(x) else None)
# day-to-day rank stability and panel coverage
stab=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):stab.append(float(q))
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except Exception: pass
# Admission requires each effective factor's actual signal panel, no proxy correlations.
evidence={}; complete=True; mx=0.; most=None
for fid in active:
 key=fid
 for prefix in ('miner_1_','miner_2_','miner_3_'): key=key.replace(prefix,'')
 paths=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not paths:
  evidence[fid]={'rho':None,'common_signal_cells':0,'file':None}; complete=False; continue
 p=max(paths,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('candidate'),L.stack().rename('library')],axis=1).dropna()
  q=spearmanr(z.candidate,z.library).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 evidence[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q): complete=False
 elif abs(q)>mx: mx=abs(float(q));most=fid
print('FACTOR',FID)
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(stab)),'implied_rank_turnover',float(1-np.mean(stab)))
print('DECAY',json.dumps({str(k):v for k,v in M.items()},sort_keys=True))
print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(evidence,sort_keys=True))
F.to_pickle('scripts/miner_2_20330317_copper_regime_conditioned_inflation_sensitivity_10x60obs_signal.pkl')

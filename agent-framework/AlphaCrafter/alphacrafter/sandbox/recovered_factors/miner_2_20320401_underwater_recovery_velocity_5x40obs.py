"""miner_2 research: underwater recovery velocity, using only daily closes.
A high score identifies an asset rebounding rapidly over five sessions while still
meaningfully below its trailing 40-session peak; it tests recovery continuation.
"""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-03-31')
def load(a):
 p='../persistent/stock_data/'+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=C.pct_change()
# Rebound over 5 sessions scaled by the absolute current 40-session underwater depth.
# Small floor prevents a near-high price from generating a mechanically huge signal.
uw=1-C/C.rolling(40,min_periods=30).max()
F=(C/C.shift(5)-1)/(uw.clip(lower=.01))
F=F.where(uw>=.01).loc[:END]
def metric(X,h):
 y=(C.shift(-h)/C-1).reindex(X.index); out=[]; ns=[]
 for d in X.index:
  z=pd.concat([X.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): out.append((d,float(q)));ns.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for sign,X in [('recovery_continuation',F),('recovery_reversal',-F)]:
 for h in [1,5,10,20]: print('HORIZON',sign,h,json.dumps(metric(X,h)[1],sort_keys=True))
 s,_=metric(X,1)
 for lab,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year.isin([2027,2028,2029,2030])),('2031_2032',s.index.year>=2031),('recent_6m',s.index>=END-pd.Timedelta(days=183))]:
  q=s[mask]; print('REGIME',sign,lab,len(q),None if len(q)==0 else float(q.mean()),None if len(q)<2 else float(q.mean()/q.std(ddof=1)),None if len(q)==0 else float((q>0).mean()))
# Cross-sectional rank persistence / turnover.
rhos=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):rhos.append(q)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except: pass
files=glob.glob('scripts/*_signal.pkl'); complete=True; mx=0.; who=None; ev={}
for fid in active:
 key=fid
 # factor ids include miner prefix in most definitions but artifact naming does too
 matches=[p for p in files if key in os.path.basename(p)]
 if not matches:
  complete=False; ev[fid]={'rho':None,'common_signal_cells':0}; continue
 p=max(matches,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna()
  q=float(spearmanr(z.x,z.l).statistic) if len(z)>=8 else np.nan
 except Exception: q=np.nan;z=pd.DataFrame()
 ev[fid]={'rho':q if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
 print('LIBRARY_CORR',fid,len(z),q)
print('SUMMARY',json.dumps({'factor_id':'miner_2_underwater_recovery_velocity_5x40obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_valid_instruments':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(rhos)),'implied_rank_turnover':float(1-np.mean(rhos)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'evidence':ev},sort_keys=True))
F.to_pickle('scripts/miner_2_20320401_underwater_recovery_velocity_5x40obs_signal.pkl')

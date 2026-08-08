"""miner_2: peer-shock residual reversal.
Tests whether assets that underperform their cross-asset peer during a broad recent
negative peer shock subsequently mean revert.  The signal is only active after a
peer 5-day drawdown, avoiding unconditional residual-momentum overlap.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-05-26')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=np.log(C).diff()
peer=pd.DataFrame({a:R.drop(columns=a).mean(axis=1) for a in A})
beta=R.rolling(60,min_periods=45).cov(peer).div(peer.rolling(60,min_periods=45).var())
# Five-day residual performance, inverted: weaker-than-beta-implied assets score higher.
res5=R.rolling(5,min_periods=5).sum()-beta*peer.rolling(5,min_periods=5).sum()
# Event gate requires a genuine broad peer drawdown (below its trailing 60d 30th percentile).
p5=peer.mean(axis=1).rolling(5,min_periods=5).sum(); threshold=p5.rolling(60,min_periods=45).quantile(.30)
gate=p5.lt(threshold)
F=(-res5).where(gate, np.nan).loc[:END]
def calc(X,h):
 y=(C.shift(-h)/C-1).reindex(X.index); out=[]; ns=[]
 for d in X.index:
  z=pd.concat([X.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q):out.append((d,float(q)));ns.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd) if sd else None,'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for orient,X in [('shock_residual_reversal',F),('shock_residual_continuation',-F)]:
 for h in [1,5,10,20]: print('HORIZON',orient,h,json.dumps(calc(X,h)[1],sort_keys=True))
 s,_=calc(X,1)
 for lab,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year.isin([2027,2028,2029,2030])),('2031_2032',s.index.year>=2031),('recent_6m',s.index>=END-pd.Timedelta(days=183))]:
  q=s[mask];print('REGIME',orient,lab,len(q),None if not len(q) else float(q.mean()),None if len(q)<2 else float(q.mean()/q.std(ddof=1)),None if not len(q) else float((q>0).mean()))
r=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):r.append(q)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except: pass
complete=True; ev={}; mx=0.; who=None
for fid in active:
 m=[p for p in glob.glob('scripts/*_signal.pkl') if fid in os.path.basename(p)]
 if not m: complete=False;ev[fid]={'rho':None,'common_signal_cells':0};continue
 try:
  L=pd.read_pickle(max(m,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna();q=spearmanr(z.x,z.l).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 ev[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z)}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
 print('LIBRARY_CORR',fid,len(z),q)
print('SUMMARY',json.dumps({'factor_id':'miner_2_peer_shock_residual_reversal_5v60obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'event_dates':int(gate.loc[:END].sum()),'mean_valid_instruments':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(r)) if r else None,'implied_rank_turnover':float(1-np.mean(r)) if r else None,'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'evidence':ev},sort_keys=True))
F.to_pickle('scripts/miner_2_20320527_peer_shock_residual_reversal_5v60obs_signal.pkl')

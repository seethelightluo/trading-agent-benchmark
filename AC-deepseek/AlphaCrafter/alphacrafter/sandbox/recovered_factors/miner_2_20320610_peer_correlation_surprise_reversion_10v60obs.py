"""miner_2: peer-correlation dispersion reversion (10/60 observations).
Tests whether assets whose short-run correlation to the rest of the cross-asset
universe is unusually low/high relative to their medium-run correlation subsequently
revert in relative return.  The signal is the negative standardized correlation change.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-06-09')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=np.log(C).diff()
# For every date/asset, estimate correlation to an equal-weight peer portfolio
# excluding that asset. Standardize its 10-session correlation surprise versus 60 sessions.
P=pd.DataFrame({a:R.drop(columns=a).mean(axis=1) for a in A})
c10=pd.DataFrame({a:R[a].rolling(10,min_periods=8).corr(P[a]) for a in A})
c60=pd.DataFrame({a:R[a].rolling(60,min_periods=45).corr(P[a]) for a in A})
sd60=c10.rolling(60,min_periods=45).std()
F=(-(c10-c60)/(sd60+1e-8)).loc[:END]
def ic(X,h):
 y=(C.shift(-h)/C-1).reindex(X.index); out=[]; ns=[]
 for d in X.index:
  z=pd.concat([X.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): out.append((d,float(q)));ns.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for name,X in [('correlation_surprise_reversion',F),('correlation_surprise_continuation',-F)]:
 for h in [1,5,10,20]: print('HORIZON',name,h,json.dumps(ic(X,h)[1],sort_keys=True))
 s,_=ic(X,1)
 for lab,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year.isin([2027,2028,2029,2030])),('2031_2032',s.index.year>=2031),('recent_6m',s.index>=END-pd.Timedelta(days=183))]:
  q=s[mask];print('REGIME',name,lab,len(q),None if not len(q) else float(q.mean()),None if len(q)<2 else float(q.mean()/q.std(ddof=1)),None if not len(q) else float((q>0).mean()))
r=[]
for n in range(1,len(F)):
 z=pd.concat([F.iloc[n-1],F.iloc[n]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):r.append(q)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except Exception: pass
complete=True;mx=0.;who=None;evidence={}
for fid in active:
 m=[p for p in glob.glob('scripts/*_signal.pkl') if fid in os.path.basename(p)]
 if not m: complete=False;evidence[fid]={'rho':None,'common_signal_cells':0};continue
 try:
  L=pd.read_pickle(max(m,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna();q=spearmanr(z.x,z.l).statistic if len(z)>=8 else np.nan
 except Exception: q=np.nan;z=pd.DataFrame()
 evidence[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z)}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
 print('LIBRARY_CORR',fid,len(z),q)
print('SUMMARY',json.dumps({'factor_id':'miner_2_peer_correlation_surprise_reversion_10v60obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_valid_instruments':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(r)),'implied_rank_turnover':float(1-np.mean(r)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'evidence':evidence},sort_keys=True))
F.to_pickle('scripts/miner_2_20320610_peer_correlation_surprise_reversion_10v60obs_signal.pkl')

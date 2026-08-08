"""Revalidate one candidate: VIX-stress-gated peer-correlation surprise continuation.
Uses all active factors' persisted signal panels for the mandatory complete-library audit.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-07-07')
def series(path):
 return pd.read_csv(path,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
C=pd.DataFrame({a:series('../persistent/stock_data/'+a+'.csv') for a in A})
R=np.log(C).diff(); V=series('../persistent/index_data/VIX.csv').reindex(C.index)
peer=pd.DataFrame({a:R.drop(columns=a).mean(axis=1) for a in A})
r10=pd.DataFrame({a:R[a].rolling(10,min_periods=8).corr(peer[a]) for a in A})
r60=pd.DataFrame({a:R[a].rolling(60,min_periods=45).corr(peer[a]) for a in A})
F=((r10-r60)/(r10.rolling(60,min_periods=45).std()+1e-8)).where(V>V.rolling(60,min_periods=45).median(),np.nan).loc[:END]
def ic(h):
 Y=(C.shift(-h)/C-1).reindex(F.index); out=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.f,z.y).statistic
   if np.isfinite(x):out.append((d,x));ns.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,dict(daily_paper_ic=float(s.mean()),daily_paper_icir=float(s.mean()/sd),ic_hit_ratio=float((s>0).mean()),ic_standard_error=float(sd/np.sqrt(len(s))),ic_dates=len(s),mean_valid_instruments=float(np.mean(ns)))
for h in [1,5,10,20]:print('H',h,json.dumps(ic(h)[1],sort_keys=True))
s,_=ic(10)
for label,mask in [('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year.isin([2027,2028,2029,2030])),('2031_2032',s.index.year>=2031),('recent_6m',s.index>=END-pd.Timedelta(days=183))]:
 q=s[mask]; print('REGIME',label,len(q),float(q.mean()),float(q.mean()/q.std(ddof=1)),float((q>0).mean()))
# explicit audit against exactly all admitted effective factors
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except Exception: pass
mx=-1; whom=None; complete=True
for fid in active:
 paths=[p for p in glob.glob('scripts/*signal.pkl') if fid in os.path.basename(p)]
 if not paths:
  print('CORR',fid,'MISSING');complete=False;continue
 L=pd.read_pickle(max(paths,key=os.path.getmtime)).reindex(index=F.index,columns=A)
 z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna()
 q=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 print('CORR',fid,len(z),q)
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);whom=fid
stab=[]
for t in range(1,len(F)):
 z=pd.concat([F.iloc[t-1],F.iloc[t]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):stab.append(q)
print('SUMMARY',json.dumps(dict(panel_dates=len(F),period=[str(F.index.min().date()),str(END.date())],coverage=float(F.notna().mean().mean()),stress_dates=int((V>V.rolling(60,min_periods=45).median()).reindex(F.index).sum()),rank_stability_1d=float(np.mean(stab)),implied_rank_turnover=float(1-np.mean(stab)),effective_library=len(active),correlation_evidence_complete=complete,max_abs_library_correlation=float(mx) if complete else None,most_correlated=whom),sort_keys=True))
F.to_pickle('scripts/miner_2_20320708_vix_stress_gated_peer_correlation_surprise_revalidation_signal.pkl')

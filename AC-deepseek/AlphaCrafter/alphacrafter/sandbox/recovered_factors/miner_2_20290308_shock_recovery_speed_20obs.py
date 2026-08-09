"""One candidate: shock-recovery speed after each asset's worst 20-day daily shock.
At each date locate the most negative single daily return in the preceding 20 sessions;
measure cumulative price recovery from that shock divided by the elapsed sessions. This
is an interpretable recovery-timing signal, distinct from raw drawdown magnitude.
Endpoint is the last fully observable session before 2029-03-08.
"""
import json, glob, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-03-07')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,'close'].astype(float)
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change()
# Index of most severe one-day shock in trailing window (ties choose most recent);
# signal is price rebound since that shock, per session elapsed.
def recovery_speed(close, ret):
 out=pd.Series(np.nan,index=close.index)
 for t in range(20,len(close)):
  w=ret.iloc[t-19:t+1]
  if w.notna().sum()<18: continue
  # np.nanargmin is safe after threshold; fill residual NaNs with +inf
  k=int(np.argmin(w.fillna(np.inf).values)); shock=t-19+k; age=t-shock
  if age>0 and np.isfinite(close.iloc[shock]) and close.iloc[shock]>0:
   out.iloc[t]=(close.iloc[t]/close.iloc[shock]-1.0)/age
 return out
F=pd.DataFrame({a:recovery_speed(P[a],R[a]) for a in A})
def calc(h):
 fw=P.shift(-h)/P-1; rows=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:
   rows.append((d,float(spearmanr(z.f,z.r).statistic))); ns.append(len(z))
 s=pd.Series(dict(rows),dtype=float); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
allm={}
for h in [1,5,10,20]:
 s,m=calc(h); allm[h]=m; print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==5:
  for label,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2025',s.index.year.isin([2024,2025])),('2026_2028',s.index.year.isin([2026,2027,2028])),('2029_YTD',s.index.year==2029)]:
   x=s[mask]; print('REGIME_5D',label,json.dumps({'dates':len(x),'ic':float(x.mean()) if len(x) else None,'icir':float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit':float((x>0).mean()) if len(x) else None}))
# turnover/coverage
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('PANEL',json.dumps({'signal_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'implied_turnover':float(1-np.mean(st))}))
# Mandatory complete library audit, artifact-backed. Any missing artifact is reported.
cor=[]; missing=[]
for jf in glob.glob('factors/*.json'):
 try:
  meta=json.load(open(jf)); status=meta.get('validation',{}).get('status')
  if status!='EFFECTIVE': continue
  fid=meta.get('factor_id',os.path.basename(jf)); hits=glob.glob('scripts/*'+fid+'*signal.pkl')
  if not hits: missing.append(fid); continue
  G=pd.read_pickle(hits[-1]); common=F.index.intersection(G.index); vals=[]
  for d in common:
   z=pd.concat([F.loc[d].rename('a'),G.loc[d].rename('b')],axis=1).dropna()
   if len(z)>=8 and z.a.nunique()>1 and z.b.nunique()>1: vals.append(abs(float(spearmanr(z.a,z.b).statistic)))
  if not vals: missing.append(fid+'(no_overlap)')
  else: cor.append((fid,float(np.mean(vals)),float(np.max(vals)),len(vals)))
 except Exception as e: missing.append(os.path.basename(jf)+'('+str(e)[:60]+')')
print('LIBRARY_CORRELATION',json.dumps({'effective_audited':len(cor),'missing_evidence':missing,'max_abs_library_correlation':max([x[1] for x in cor],default=None),'details':cor},default=str))
F.to_pickle('scripts/miner_2_20290308_shock_recovery_speed_20obs_signal.pkl')

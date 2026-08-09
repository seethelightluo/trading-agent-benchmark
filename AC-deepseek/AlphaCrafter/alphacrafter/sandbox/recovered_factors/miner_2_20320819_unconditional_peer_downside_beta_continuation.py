"""One candidate: unconditional peer-downside beta continuation, 40-session rolling estimate."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-08-18')
def load(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
C=pd.DataFrame({a:load('../persistent/stock_data/'+a+'.csv') for a in A}); R=np.log(C).diff()
P=pd.DataFrame({a:R.drop(columns=a).mean(axis=1) for a in A})
# For each asset, beta to equal-weight peers using only peer-negative sessions in a trailing 40-day window.
F=pd.DataFrame(index=C.index,columns=A,dtype=float)
for a in A:
 x=R[a].where(P[a]<0); y=P[a].where(P[a]<0)
 F[a]=x.rolling(40,min_periods=24).cov(y)/y.rolling(40,min_periods=24).var()
F=F.loc[:END]
def getic(h):
 Y=(C.shift(-h)/C-1).reindex(F.index); vals=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): vals.append((d,q)); ns.append(len(z))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for h in [1,5,10,20]: print('H',h,json.dumps(getic(h)[1],sort_keys=True))
s,_=getic(10)
for label,mask in [('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year.isin([2027,2028,2029,2030])),('2031_2032',s.index.year>=2031),('recent_6m',s.index>=END-pd.Timedelta(days=183))]:
 q=s[mask]; print('REGIME',label,len(q),float(q.mean()) if len(q) else None,float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,float((q>0).mean()) if len(q) else None)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except: pass
complete=True;mx=-1;who=None
for fid in active:
 ps=[p for p in glob.glob('scripts/*signal.pkl') if fid in os.path.basename(p)]
 if not ps: print('CORR',fid,'MISSING');complete=False;continue
 L=pd.read_pickle(max(ps,key=os.path.getmtime)).reindex(index=F.index,columns=A)
 z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna()
 q=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 print('CORR',fid,len(z),q)
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
st=[]
for t in range(1,len(F)):
 z=pd.concat([F.iloc[t-1],F.iloc[t]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('SUMMARY',json.dumps({'period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'rank_stability_1d':float(np.nanmean(st)),'implied_rank_turnover':float(1-np.nanmean(st)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':float(mx) if complete else None,'most_correlated':who},sort_keys=True))
F.to_pickle('scripts/miner_2_20320819_unconditional_peer_downside_beta_continuation_signal.pkl')

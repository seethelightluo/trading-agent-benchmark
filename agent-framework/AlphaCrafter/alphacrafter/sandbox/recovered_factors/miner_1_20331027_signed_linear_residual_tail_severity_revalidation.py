"""Scheduled revalidation (one existing factor idea) through latest visibility-safe close."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ends=[]; raw={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']; raw[a]=d; ends.append(d.index.max())
END=min(ends); p=pd.DataFrame(raw).loc[:END]; r=p.pct_change(); e=r.sub(r.mean(axis=1),axis=0)
print('REVALIDATION signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d cutoff',END.date(),'calendar_dates',len(p),'assets',len(A))
def tail(x,w,n):
 def zfun(z):
  s=np.std(z,ddof=1); q=z[z < -s]
  return -q.mean()/(s+1e-12) if len(q) else 0.
 return x.rolling(w,min_periods=n).apply(zfun,raw=True)
def recovery(x,w,n):
 def zfun(z):
  s=np.std(z[:-1],ddof=1)
  if not np.isfinite(s) or s<1e-12:return np.nan
  q=z[:-1] < -s
  return np.mean(z[1:][q])/s if q.any() else 0.
 return x.rolling(w,min_periods=n).apply(zfun,raw=True)
sev=pd.DataFrame({a:tail(e[a],20,14)-tail(e[a],60,42) for a in A})
rec=-pd.DataFrame({a:recovery(e[a],20,14)-recovery(e[a],60,42) for a in A})
f=(sev.rank(axis=1,pct=True)-.5)+(rec.rank(axis=1,pct=True)-.5)
print('VALID_CELLS',int(f.notna().sum().sum()),'COVERAGE',round(float(f.notna().mean().mean()),6))
ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p)-1; out=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):out.append((d,v));ns.append(len(q))
 s=pd.Series(dict(out),dtype=float);ics[h]=s; sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.4f se=%.6f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns),sd/np.sqrt(len(s))))
for nm,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_onward','2027-01-01',str(END.date()))]:
 s=ics[10].loc[lo:hi]; print('REGIME10',nm,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6) if len(s)>1 else None,'hit',round((s>0).mean(),6))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('TURNOVER',round(float(np.mean(ts)),6),'pairs',len(ts))
print('DECAY', {h:(round(s.mean(),6),round(s.mean()/s.std(ddof=1),6),len(s)) for h,s in ics.items()})
f.to_pickle('scripts/miner_1_20331027_signed_linear_residual_tail_severity_revalidation_signal.pkl')
# Evidence is complete only if every currently EFFECTIVE definition has a reusable signal artifact.
active=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn));
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except Exception: pass
rows=[]; missing=[]
for fid in active:
 hits=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not hits: missing.append(fid); continue
 x=pd.read_pickle(max(hits,key=os.path.getmtime)); vals=[]; cells=0
 for d in f.index.intersection(x.index):
  q=pd.concat([f.loc[d].rename('a'),x.loc[d].rename('b')],axis=1).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1:
   v=spearmanr(q.a,q.b).statistic
   if np.isfinite(v):vals.append(abs(v)); cells+=len(q)
 if vals: rows.append((max(vals),fid,len(vals),cells))
 else: missing.append(fid)
print('LIBRARY_AUDIT active',len(active),'compared',len(rows),'missing_evidence',len(missing))
print('MISSING_IDS',','.join(missing[:8]))
if rows: print('MAX_AVAILABLE_RHO',max(rows))
print('LIBRARY_EVIDENCE_COMPLETE',not missing)

"""Miner 1: residual peer-linkage decoupling factor, 20d versus 60d.
Higher signal = an asset's correlation to its ex-self equal-weight peer basket has fallen
relative to its longer baseline, interpreted as improving diversification/relative resilience.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-08-31')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change()
def peer_corr(w):
 out=pd.DataFrame(index=r.index,columns=A,dtype=float)
 for a in A:
  peer=r[[b for b in A if b!=a]].mean(axis=1)
  out[a]=r[a].rolling(w,min_periods=int(.75*w)).corr(peer)
 return out
f=(peer_corr(60)-peer_corr(20)).replace([np.inf,-np.inf],np.nan)
print('CANDIDATE peer_linkage_decoupling_20_60 cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',f.dropna(how='all').shape[0],'valid_cells',int(f.notna().sum().sum()),'coverage',round(float(f.notna().mean().mean()),6))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; rows=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): rows.append((d,z)); ns.append(len(q))
 s=pd.Series(dict(rows),dtype=float); ics[h]=s; sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for nm,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=s.loc[lo:hi]; print('REGIME10',nm,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(turns)),6),'pairs',len(turns))
print('DECAY',{h:(round(float(s.mean()),6),round(float(s.mean()/s.std(ddof=1)),6),len(s)) for h,s in ics.items()})
# Binding independence check: only effective persisted definitions, and each must have an artifact.
active=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except Exception: pass
mx=-1; who=None; evidence=[]; missing=[]
for fid in active:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits: missing.append(fid); continue
 x=pd.read_pickle(max(hits,key=os.path.getmtime)); vals=[]; nobs=[]
 for d in f.index.intersection(x.index):
  q=pd.concat([f.loc[d].rename('f'),x.loc[d].reindex(A).rename('x')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.x.nunique()>1:
   z=spearmanr(q.f,q.x).statistic
   if np.isfinite(z): vals.append(abs(z));nobs.append(len(q))
 if not vals: missing.append(fid); continue
 z=max(vals); evidence.append((fid,z,len(vals),np.mean(nobs)))
 if z>mx: mx,who=z,fid
print('ACTIVE_LIBRARY',len(active),'evidence',len(evidence),'missing',len(missing))
for e in sorted(evidence,key=lambda q:-q[1])[:8]: print('COMPARE',e[0],'maxrho=%.6f dates=%d meanN=%.2f'%(e[1],e[2],e[3]))
print('MAX_ABS_LIBRARY_CORRELATION %.6f %s'%(mx,who))
f.to_pickle('scripts/miner_1_20330901_peer_linkage_decoupling_20_60_signal.pkl')

"""Miner_1: residual drawdown-compression / recovery-slope factor; one path-shape idea."""
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-08-03')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change(); resid=r.sub(r.mean(axis=1),axis=0)
# Within each 20d residual path, quantify improvement in drawdown: mean drawdown
# over most recent 10 days less severe than the first 10; combine with recent residual slope.
def pathstat(x):
 x=np.asarray(x,float)
 if not np.isfinite(x).all(): return np.nan
 c=np.cumprod(1+x); dd=c/np.maximum.accumulate(c)-1
 old=dd[:10].mean(); new=dd[10:].mean()
 slope=np.polyfit(np.arange(10),c[10:],1)[0]/np.mean(c[10:])
 # Both components increase when residual path is recovering / compressing drawdown.
 return (new-old)+slope*10
f=resid.rolling(20,min_periods=20).apply(pathstat,raw=True).replace([np.inf,-np.inf],np.nan)
print('CANDIDATE residual_drawdown_compression_recovery_slope_20d cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',int(f.notna().any(axis=1).sum()),'valid_cells',int(f.notna().sum().sum()),'coverage',round(float(f.notna().mean().mean()),6))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; vals=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z));ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float);ics[h]=s; sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for nm,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=s.loc[lo:hi]; print('REGIME10',nm,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(turn)),6),'pairs',len(turn))
print('DECAY',{h:(round(float(s.mean()),6),round(float(s.mean()/s.std(ddof=1)),6),len(s)) for h,s in ics.items()})
# Compare to every available factor signal artifact; evidence only valid if all artifacts load.
maxrho=-1.; maxname=None; compared=0; errors=[]
for fn in glob.glob('scripts/*_signal.pkl'):
 try:
  x=pd.read_pickle(fn)
  if not isinstance(x,pd.DataFrame): raise ValueError('not dataframe')
  ds=f.index.intersection(x.index);cs=f.columns.intersection(x.columns); vals=[]
  for d in ds:
   q=pd.concat([f.loc[d,cs].rename('f'),x.loc[d,cs].rename('x')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1 and q.x.nunique()>1:
    z=spearmanr(q.f,q.x).statistic
    if np.isfinite(z):vals.append(abs(z))
  if not vals: raise ValueError('no common IC-sized cross sections')
  compared+=1; z=max(vals)
  if z>maxrho:maxrho,maxname=z,os.path.basename(fn)
 except Exception as e: errors.append((os.path.basename(fn),str(e)))
print('LIBRARY_CORRELATION max_abs=%.6f artifact=%s artifacts=%d errors=%d'%(maxrho,maxname,compared,len(errors)))
if errors: print('LIBRARY_ERRORS',errors[:10])
f.to_pickle('scripts/miner_1_20330804_residual_drawdown_compression_recovery_slope_20d_signal.pkl')

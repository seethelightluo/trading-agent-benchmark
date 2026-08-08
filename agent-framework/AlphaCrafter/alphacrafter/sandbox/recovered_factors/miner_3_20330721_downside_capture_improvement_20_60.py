"""Miner_3: downside-capture transition vs peer basket; one interpretable tail-transition idea."""
import os,glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-07-20')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change()
# For each asset, estimate beta to the equal-weight peer basket only on peer-down days.
# Signal rewards a recent reduction in downside capture relative to its 60d baseline.
def downbeta(w):
 out=pd.DataFrame(index=r.index,columns=A,dtype=float)
 for a in A:
  peer=r[[x for x in A if x!=a]].mean(axis=1)
  mask=peer<0
  x=peer.where(mask); y=r[a].where(mask)
  cov=y.rolling(w,min_periods=max(12,int(.60*w))).cov(x)
  var=x.rolling(w,min_periods=max(12,int(.60*w))).var()
  out[a]=cov/var.replace(0,np.nan)
 return out
b20=downbeta(20); b60=downbeta(60)
f=(-(b20-b60)).replace([np.inf,-np.inf],np.nan)
print('CANDIDATE downside_capture_improvement_20_60 cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',int(f.notna().any(axis=1).sum()),'valid_cells',int(f.notna().sum().sum()),'coverage',round(float(f.notna().mean().mean()),6))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1;vals=[];ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z):vals.append((d,z));ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float);ics[h]=s;sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for nm,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=s.loc[lo:hi];print('REGIME10',nm,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(turn)),6),'pairs',len(turn))
print('DECAY',{h:(round(float(s.mean()),6),round(float(s.mean()/s.std(ddof=1)),6),len(s)) for h,s in ics.items()})
maxrho=-1;maxname=None;compared=0
for fn in glob.glob('scripts/*_signal.pkl'):
 try:
  x=pd.read_pickle(fn)
  if not isinstance(x,pd.DataFrame):continue
  ds=f.index.intersection(x.index); cs=f.columns.intersection(x.columns); vals=[]
  for d in ds:
   q=pd.concat([f.loc[d,cs].rename('f'),x.loc[d,cs].rename('x')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1 and q.x.nunique()>1:
    z=spearmanr(q.f,q.x).statistic
    if np.isfinite(z): vals.append(abs(z))
  if vals:
   compared+=1;z=max(vals)
   if z>maxrho:maxrho,maxname=z,os.path.basename(fn)
 except Exception:pass
print('LIBRARY_CORRELATION max_abs=%.6f artifact=%s artifacts=%d'%(maxrho,maxname,compared))
f.to_pickle('scripts/miner_3_20330721_downside_capture_improvement_20_60_signal.pkl')

"""One candidate: residual downside-volume absorption (20d).
For each asset, measure whether volume is relatively heavier on down days than
up days over the last 20 completed observations, then residualize that measure
cross-sectionally against 20-day return and 20-day volatility.  High values
indicate that an asset has absorbed selling participation beyond its own recent
trend/risk, a potentially defensive path-quality signal."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def field(a,x):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 return d[x]
close=pd.DataFrame({a:field(a,'close') for a in A}); volume=pd.DataFrame({a:field(a,'volume') for a in A})
end=close.dropna(how='any').index.max(); ix=pd.date_range(close.index.min(),end,freq='B')
p=close.reindex(ix).ffill(); v=volume.reindex(ix).ffill(); r=p.pct_change(); lv=np.log(v.where(v>0))
# Difference in mean log-volume for negative versus positive return days,
# requiring at least four observations in each state to avoid thin estimates.
down=(r<0); up=(r>0)
dsum=(lv*down).rolling(20,min_periods=20).sum(); usum=(lv*up).rolling(20,min_periods=20).sum()
dn=down.rolling(20,min_periods=20).sum(); un=up.rolling(20,min_periods=20).sum()
imb=dsum/dn-usum/un; imb[(dn<4)|(un<4)]=np.nan
r20=p.pct_change(20); vol20=r.rolling(20,min_periods=20).std()
f=pd.DataFrame(np.nan,index=ix,columns=A)
for d in ix:
 q=pd.concat([imb.loc[d].rename('imb'),r20.loc[d].rename('r20'),vol20.loc[d].rename('vol')],axis=1).dropna()
 if len(q)>=8 and q.nunique().min()>1:
  X=np.c_[np.ones(len(q)),q.r20,q.vol]
  f.loc[d,q.index]=q.imb-X@np.linalg.lstsq(X,q.imb,rcond=None)[0]
print('FACTOR residual_downside_volume_absorption_20 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in ix:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z));ns.append(len(q))
 s=pd.Series(dict(vals));ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for h in [5,10]:
 for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01','2034-12-31'),('2035','2035-01-01',end)]:
  s=ics[h].loc[lo:hi]; print('REGIME H%d %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(h,nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(turn),len(turn),z.abs().stack().mean()))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE': eff.append(j['factor_id'])
 except Exception: pass
peers=[];missing=[]
for fid in eff:
 paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not paths: missing.append(fid);continue
 old=pd.read_pickle(max(paths,key=os.path.getmtime)); old=old.get('signal',old) if isinstance(old,dict) else old
 if not isinstance(old,pd.DataFrame): missing.append(fid);continue
 q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2:missing.append(fid);continue
 peers.append((abs(spearmanr(q.x,q.z).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(peers),len(missing),('%.6f'%max(x[0] for x in peers) if len(peers)==len(eff) else 'UNAVAILABLE')))
if peers: print('MAX_OBSERVED rho=%.6f factor=%s cells=%d'%max(peers))
if missing: print('MISSING',','.join(missing))
f.to_pickle('scripts/miner_1_20350607_residual_downside_volume_absorption_20_signal.pkl')

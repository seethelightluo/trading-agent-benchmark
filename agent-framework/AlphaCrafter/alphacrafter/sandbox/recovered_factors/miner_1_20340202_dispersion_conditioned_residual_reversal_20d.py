"""One candidate: cross-sectionally dispersion-conditioned 20d residual reversal, cutoff 2034-02-01."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2034-02-01')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change()
# Remove common daily cross-asset move.  A negative 20d idiosyncratic return is rewarded,
# but only in proportion to contemporaneous cross-sectional dispersion: reversals should be
# more meaningful when assets have actually diverged.
eps=r.sub(r.median(axis=1),axis=0)
res20=(1+eps).rolling(20,min_periods=16).apply(np.prod,raw=True)-1
disp=eps.std(axis=1).rolling(20,min_periods=16).mean()
scale=(disp/disp.rolling(252,min_periods=126).median()).clip(0.5,2.0)
f=(-res20).mul(scale,axis=0).replace([np.inf,-np.inf],np.nan)
print('FACTOR dispersion_conditioned_residual_reversal_20d VALIDATED_THROUGH',CUT.date())
print('expression=-((product_20(1+(r_i-median_j(r_j))))-1)*clip(mean_20(cs_std(r))/median_252(mean_20(cs_std(r))),0.5,2.0)')
print('coverage=%.6f calendar_dates=%d valid_cells=%d assets=%d'%(f.notna().mean().mean(),f.notna().any(axis=1).sum(),f.notna().sum().sum(),len(A)))
stats={}
for h in [1,5,10,20]:
 vals=[];ns=[];fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z):vals.append((d,z));ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float);stats[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for name,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2034','2031-01-01',CUT)]:
 s=stats[10].loc[lo:hi];print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(name,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
  z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(z):turn.append(1-z)
print('turnover=%.6f pairs=%d'%(np.mean(turn),len(turn)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  x=json.load(open(fn))
  if x.get('validation',{}).get('status')=='EFFECTIVE':eff.append(x['factor_id'])
 except:pass
found=[];scores=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if hits:
  found.append(fid);z=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1:scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE artifacts=%d effective=%d missing=%d'%(len(found),len(eff),len(eff)-len(found)))
print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION='+('%.6f factor=%s cells=%d'%max(scores) if scores else 'UNAVAILABLE'))
f.to_pickle('scripts/miner_1_20340202_dispersion_conditioned_residual_reversal_20d_signal.pkl')

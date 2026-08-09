"""One idea: inverse residual serial-persistence acceleration (5d vs 20d)."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF=pd.Timestamp('2034-06-21')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:load(a) for a in A}).loc[:CUTOFF]; CUT=p.dropna(how='all').index.max(); p=p.loc[:CUT]
r=p.pct_change(); m=r.mean(axis=1); v=m.rolling(60,min_periods=40).var().replace(0,np.nan)
beta=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(m).div(v) for a in A})
e=r-beta.mul(m,axis=0)
# Each term is a rolling lag-one correlation of idiosyncratic daily returns.
# Negative acceleration identifies assets whose recent residual persistence has fallen
# below its medium-window baseline, a simple potential mean-reversion transition.
def ac(s,w): return s.rolling(w,min_periods=w).corr(s.shift(1))
f=-(pd.DataFrame({a:ac(e[a],5)-ac(e[a],20) for a in A})).replace([np.inf,-np.inf],np.nan)
print('FACTOR inverse_residual_serial_persistence_acceleration_5_20_60d VALIDATED_THROUGH',CUT.date())
print('definition= - [rolling_corr_lag1(residual_return,5) - rolling_corr_lag1(residual_return,20)], residual=return-60d beta*equal-weight market return')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1; vals=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   x=spearmanr(q.f,q.y).statistic
   if np.isfinite(x): vals.append((d,x));ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float);ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01',CUT)]:
 s=ics[5].loc[lo:hi];print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(ts),len(ts)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn))
  if z.get('validation',{}).get('status')=='EFFECTIVE':eff.append(z['factor_id'])
 except Exception:pass
scores=[];missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits:missing.append(fid);continue
 old=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),old.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
if len(scores)==len(eff): print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else: print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
f.to_pickle('scripts/miner_3_20340622_inverse_residual_serial_persistence_acceleration_5_20_60d_signal.pkl')

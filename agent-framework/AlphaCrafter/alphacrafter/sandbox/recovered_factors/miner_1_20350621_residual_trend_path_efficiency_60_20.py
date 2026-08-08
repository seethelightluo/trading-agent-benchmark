"""One price-only candidate: residual trend-path efficiency, 60/20d.
For each asset, 60d net return divided by root-sum-square daily returns measures
whether its move was smooth rather than jumpy. Daily cross-sectional residuals
against 60d return and 20d realized volatility isolate path quality from trend
and ordinary risk. Forward returns are validation-only."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
raw=pd.DataFrame({a:load(a) for a in A})
end=raw.dropna(how='any').index.max()
p=raw.loc[:end].reindex(pd.date_range(raw.index.min(),end,freq='B')).ffill()
r=p.pct_change()
# Efficiency normalized by RSS, then cross-sectionally residualize contemporaneous return and risk.
eff=p.pct_change(60).div(np.sqrt(r.pow(2).rolling(60,min_periods=60).sum()))
vol=r.rolling(20,min_periods=20).std()
f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for d in p.index:
 q=pd.DataFrame({'y':eff.loc[d],'trend':p.pct_change(60).loc[d],'vol':vol.loc[d]}).dropna()
 if len(q)>=8:
  X=np.c_[np.ones(len(q)),q.trend,q.vol]
  f.loc[d,q.index]=q.y-np.linalg.lstsq(X,q.y,rcond=None)[0].dot(X.T)
print('FACTOR residual_trend_path_efficiency_60_20 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in (1,5,10,20):
 fw=p.shift(-h).div(p)-1; vals=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z));ns.append(len(q))
 s=pd.Series(dict(vals)); ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for label,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_29','2025-01-01','2029-12-31'),('2030_34','2030-01-01','2034-12-31'),('2035','2035-01-01',str(end.date()))]:
 s=ics[5].loc[lo:hi];print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(label,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(turns),len(turns),z.abs().stack().mean()))
effids=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE':effids.append(j['factor_id'])
 except Exception:pass
scores=[];missing=[];peers=[]
for fid in effids:
 paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not paths:missing.append(fid);continue
 old=pd.read_pickle(max(paths,key=os.path.getmtime)); old=old.get('signal',old) if isinstance(old,dict) else old
 q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2:missing.append(fid);continue
 rho=abs(spearmanr(q.x,q.z).statistic);scores.append(rho);peers.append((rho,fid,len(q)))
mx='%.6f'%max(scores) if len(scores)==len(effids) and scores else 'UNAVAILABLE'
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(effids),len(scores),len(missing),mx))
if peers: print('MAX_OBSERVED rho=%.6f factor=%s cells=%d'%max(peers))
f.to_pickle('scripts/miner_1_20350621_residual_trend_path_efficiency_60_20_signal.pkl')
"""
# strip accidental enclosing triple? 
content=content[3:-4] if content.startswith('"""') else content
# Above manipulation leaves script docstring start absent; write direct reconstructed content
write_file('scripts/miner_1_20350621_residual_trend_path_efficiency_60_20.py',content)
print(len(content))
PY
python scripts/miner_1_20350621_residual_trend_path_efficiency_60_20.py
"""One OHLC factor: residual downside intraday-recovery confirmation, 5/20d.
Assets that had weak five-day performance but repeatedly close high within their
daily ranges may exhibit a short-horizon reversal.  The factor is residualized
against ordinary 20d return, 20d volatility and 20d range to separate recovery
from trend/risk/range effects."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def col(a,c):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()[c]
C=pd.DataFrame({a:col(a,'close') for a in A}); H=pd.DataFrame({a:col(a,'high') for a in A}); L=pd.DataFrame({a:col(a,'low') for a in A})
end=C.dropna(how='any').index.max(); ix=pd.date_range(C.index.min(),end,freq='B')
C=C.reindex(ix).ffill(); H=H.reindex(ix).ffill(); L=L.reindex(ix).ffill(); R=C.pct_change()
# High close location, only meaningful after a negative 5d move; preserves strength of both legs.
loc=((C-L)/(H-L).replace(0,np.nan)).clip(0,1)
raw=(-C.pct_change(5))*loc.rolling(5,min_periods=5).mean()
ret20=C.pct_change(20); vol20=R.rolling(20,min_periods=20).std(); rng20=((H-L)/C.shift(1)).rolling(20,min_periods=20).mean()
f=pd.DataFrame(index=ix,columns=A,dtype=float)
for d in ix:
 q=pd.DataFrame({'y':raw.loc[d],'ret':ret20.loc[d],'vol':vol20.loc[d],'rng':rng20.loc[d]}).dropna()
 if len(q)>=8:
  X=np.c_[np.ones(len(q)),q[['ret','vol','rng']]]
  f.loc[d,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
print('FACTOR residual_downside_intraday_recovery_confirmation_5_20 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in (1,5,10,20):
 fw=C.shift(-h)/C-1; out=[]; ns=[]
 for d in ix:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):out.append((d,v));ns.append(len(q))
 s=pd.Series(dict(out));ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for label,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_29','2025-01-01','2029-12-31'),('2030_34','2030-01-01','2034-12-31'),('2035','2035-01-01',str(end.date()))]:
 s=ics[5].loc[lo:hi];print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(label,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(ts),len(ts),z.abs().stack().mean()))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn));
  if j.get('validation',{}).get('status')=='EFFECTIVE':eff.append(j['factor_id'])
 except:pass
scores=[]; missing=[]; peers=[]
for fid in eff:
 paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not paths:missing.append(fid);continue
 old=pd.read_pickle(max(paths,key=os.path.getmtime));old=old.get('signal',old) if isinstance(old,dict) else old
 q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2:missing.append(fid);continue
 rho=abs(spearmanr(q.x,q.z).statistic);scores.append(rho);peers.append((rho,fid,len(q)))
mx=max(scores) if len(scores)==len(eff) and scores else None
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(scores),len(missing),'%.6f'%mx if mx is not None else 'UNAVAILABLE'))
if peers:print('MAX_OBSERVED rho=%.6f factor=%s cells=%d'%max(peers))
f.to_pickle('scripts/miner_1_20350719_residual_downside_intraday_recovery_confirmation_5_20_signal.pkl')

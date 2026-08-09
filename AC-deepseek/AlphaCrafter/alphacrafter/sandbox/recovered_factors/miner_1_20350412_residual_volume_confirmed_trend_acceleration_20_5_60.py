"""One idea: residual volume-confirmed trend acceleration (20/5/60).
The raw score is 20-day return times the log change of mean volume from 60 to
5 days.  Each day it is cross-sectionally residualized on ordinary 20-day
return and 20-day realised volatility, retaining unusual volume confirmation
rather than generic trend or risk.  All inputs end on the signal date."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2035-04-11')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 return d.loc[:CUT,['close','volume']]
x={a:load(a) for a in A}
close=pd.DataFrame({a:x[a]['close'] for a in A}); volume=pd.DataFrame({a:x[a]['volume'] for a in A})
end=close.dropna(how='any').index.max(); idx=pd.date_range(close.index.min(),end,freq='B')
p=close.loc[:end].reindex(idx).ffill(); v=volume.loc[:end].reindex(idx).ffill()
r=p.pct_change(); r20=p.pct_change(20); rv20=r.rolling(20,min_periods=20).std()
# Volume is only used where positive and observed; no artificial imputation beyond a closed-date fill.
v5=v.rolling(5,min_periods=5).mean(); v60=v.rolling(60,min_periods=60).mean()
raw=r20*np.log(v5.where(v5>0)/v60.where(v60>0))
f=pd.DataFrame(np.nan,index=idx,columns=A)
for d in idx:
 q=pd.concat([raw.loc[d].rename('raw'),r20.loc[d].rename('r20'),rv20.loc[d].rename('rv')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8 and q.nunique().min()>1:
  X=np.column_stack([np.ones(len(q)),q.r20.values,q.rv.values])
  f.loc[d,q.index]=q.raw.values-X@np.linalg.lstsq(X,q.raw.values,rcond=None)[0]
print('FACTOR residual_volume_confirmed_trend_acceleration_20_5_60 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in idx:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z));ns.append(len(q))
 s=pd.Series(dict(vals)); ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01','2034-12-31'),('2035','2035-01-01',end)]:
 s=ics[5].loc[lo:hi]
 print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(turns),len(turns),z.abs().stack().mean()))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE': eff.append(j['factor_id'])
 except Exception: pass
scores=[]; missing=[]; peers=[]
for fid in eff:
 paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not paths: missing.append(fid); continue
 old=pd.read_pickle(max(paths,key=os.path.getmtime)); old=old.get('signal',old) if isinstance(old,dict) else old
 q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2: missing.append(fid); continue
 rho=abs(spearmanr(q.x,q.z).statistic); scores.append(rho); peers.append((rho,fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(scores),len(missing),('%.6f'%max(scores) if len(scores)==len(eff) and scores else 'UNAVAILABLE')))
if peers: print('MAX_OBSERVED rho=%.6f factor=%s cells=%d'%max(peers))
f.to_pickle('scripts/miner_1_20350412_residual_volume_confirmed_trend_acceleration_20_5_60_signal.pkl')

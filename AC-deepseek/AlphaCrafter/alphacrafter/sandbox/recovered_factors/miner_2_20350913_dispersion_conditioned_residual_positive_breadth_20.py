"""One candidate: dispersion-conditioned residual positive-day breadth (20 sessions).
The signal is residual positive-day breadth, activated only when lagged 20-day
cross-asset return dispersion exceeds its prior 60-session median.  This tests
whether persistent relative leadership is useful specifically when the market
is sufficiently dispersed for cross-sectional ranks to be meaningful.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
raw=pd.DataFrame({a:load(a) for a in A}); end=raw.dropna(how='any').index.max(); raw=raw.loc[:end]
p=raw.reindex(pd.date_range(raw.index.min(),end,freq='B')).ffill(); r=p.pct_change(); resid=r.sub(r.mean(axis=1),axis=0)
breadth=(resid>0).where(resid.notna()).rolling(20,min_periods=20).mean()
# Decision uses only returns ending yesterday: shift both condition inputs one row.
disp=r.std(axis=1); trailing_disp=disp.rolling(20,min_periods=20).mean().shift(1)
threshold=trailing_disp.rolling(60,min_periods=60).median().shift(1)
active=trailing_disp.gt(threshold)
f=breadth.where(active, np.nan)
print('FACTOR dispersion_conditioned_residual_positive_breadth_20 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f active_dates=%d active_rate=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),active.sum(),active.mean()))
ics={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v): vals.append((d,v));ns.append(len(q))
 s=pd.Series(dict(vals));ics[h]=s
 print('H%d daily_paper_IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2033','2027-01-01','2033-12-31'),('2034_current','2034-01-01',end)]:
 s=ics[10].loc[lo:hi]; print('REGIME_H10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
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
scores=[];missing=[]
for fid in eff:
 paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not paths:missing.append(fid);continue
 old=pd.read_pickle(max(paths,key=os.path.getmtime));old=old.get('signal',old) if isinstance(old,dict) else old
 if not isinstance(old,pd.DataFrame):missing.append(fid);continue
 q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2:missing.append(fid)
 else:scores.append(abs(spearmanr(q.x,q.z).statistic))
mx=max(scores) if len(scores)==len(eff) and scores else None
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(scores),len(missing),'%.6f'%mx if mx is not None else 'UNAVAILABLE'))
f.to_pickle('scripts/miner_2_20350913_dispersion_conditioned_residual_positive_breadth_20_signal.pkl')

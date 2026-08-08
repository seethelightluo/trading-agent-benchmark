"""One candidate: volume-confirmed intermediate trend (20d return, 5/60 volume ratio).
A positive 20-session trend receives a larger score only if recent activity is
above the asset's own trailing normal volume. This uses each asset's relative
volume, avoiding incomparable volume units across assets. All inputs lag one
completed session.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def field(a,x):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()[x]
rawp=pd.DataFrame({a:field(a,'close') for a in A}); end=rawp.dropna(how='any').index.max(); rawp=rawp.loc[:end]
rawv=pd.DataFrame({a:field(a,'volume') for a in A}).reindex(rawp.index)
p=rawp.reindex(pd.date_range(rawp.index.min(),end,freq='B')).ffill()
v=rawv.reindex(p.index).ffill()
# Require genuinely positive volume; zero-volume benchmark series are deliberately not inferred.
volratio=(v.rolling(5,min_periods=4).mean()/v.rolling(60,min_periods=40).median()).where(v.rolling(60,min_periods=40).median()>0)
trend=p.pct_change(20)
f=(trend*np.log(volratio.clip(lower=.20,upper=5))).shift(1).replace([np.inf,-np.inf],np.nan)
print('FACTOR volume_confirmed_trend_20_5v60 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f mean_valid_per_date=%.3f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),f.notna().sum(axis=1).mean()))
ics={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z)); ns.append(len(q))
 s=pd.Series(dict(vals));ics[h]=s
 print('H%d daily_paper_IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2033','2027-01-01','2033-12-31'),('2034_current','2034-01-01',end)]:
 s=ics[10].loc[lo:hi];print('REGIME_H10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:ts.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
z=f.sub(f.mean(axis=1),axis=0).div(f.std(axis=1).replace(0,np.nan),axis=0)
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(ts),len(ts),z.abs().stack().mean()))
# Mandatory complete library evidence check.
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE': eff.append(j['factor_id'])
 except Exception: pass
scores=[]; missing=[]
for fid in eff:
 paths=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not paths: missing.append(fid); continue
 old=pd.read_pickle(max(paths,key=os.path.getmtime)); old=old.get('signal',old) if isinstance(old,dict) else old
 if not isinstance(old,pd.DataFrame): missing.append(fid); continue
 q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2: missing.append(fid)
 else: scores.append(abs(spearmanr(q.x,q.z).statistic))
mx=max(scores) if len(scores)==len(eff) and scores else None
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(scores),len(missing),'%.6f'%mx if mx is not None else 'UNAVAILABLE'))
f.to_pickle('scripts/miner_2_20351025_volume_confirmed_trend_20_5v60_signal.pkl')

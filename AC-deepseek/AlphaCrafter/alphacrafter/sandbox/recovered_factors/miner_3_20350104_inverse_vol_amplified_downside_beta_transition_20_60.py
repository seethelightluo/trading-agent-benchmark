"""One idea: volatility-amplified downside-beta transition (20d versus 60d)."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2035-01-03')
def load(a):
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
    return x.loc[:CUT]
raw=pd.DataFrame({a:load(a) for a in A})
end=raw.dropna(how='any').index.max()
p=raw.loc[:end].reindex(pd.date_range(raw.index.min(),end,freq='B')).ffill()
r=p.pct_change(); m=r.median(axis=1); down=m.where(m<0)
def dbeta(w):
    return r.mul(down,axis=0).rolling(w,min_periods=max(8,w//4)).sum().div(down.pow(2).rolling(w,min_periods=max(8,w//4)).sum().replace(0,np.nan),axis=0)
# Negative transition favors assets whose downside-market exposure is contracting; amplification
# assigns greater importance when the asset's own recent volatility is rising versus its baseline.
trans=dbeta(20)-dbeta(60)
vol20=r.rolling(20,min_periods=10).std(); vol60=r.rolling(60,min_periods=30).std()
f=-trans*vol20.div(vol60.replace(0,np.nan))
print('FACTOR inverse_vol_amplified_downside_beta_transition_20_60 VALIDATED_THROUGH',end.date())
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v): vals.append((d,v)); ns.append(len(q))
 s=pd.Series(dict(vals)); ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01',end)]:
 s=ics[5].loc[lo:hi]
 print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
# concentration is mean cross-sectional absolute z-score; 0 implies a completely flat signal.
z=(f-f.mean(axis=1).values[:,None]).div(f.std(axis=1).replace(0,np.nan).values[:,None])
print('turnover=%.6f pairs=%d concentration_abs_z=%.6f'%(np.mean(turns),len(turns),z.abs().stack().mean()))
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
 old=pd.read_pickle(max(paths,key=os.path.getmtime)); q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.z.nunique()<2: missing.append(fid)
 else: scores.append(abs(spearmanr(q.x,q.z).statistic))
print('INDEPENDENCE effective=%d evidence=%d missing=%d max_abs_library_correlation=%s'%(len(eff),len(scores),len(missing),('%.6f'%max(scores) if len(scores)==len(eff) and scores else 'UNAVAILABLE')))
f.to_pickle('scripts/miner_3_20350104_inverse_vol_amplified_downside_beta_transition_20_60_signal.pkl')

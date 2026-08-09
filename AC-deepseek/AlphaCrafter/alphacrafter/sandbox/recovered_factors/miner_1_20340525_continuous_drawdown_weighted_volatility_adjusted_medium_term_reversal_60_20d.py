"""One idea: continuously drawdown-weighted volatility-adjusted medium-term reversal.
Uses only close data through the prior completed daily bar."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF=pd.Timestamp('2034-05-24')
def close(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:close(a) for a in A}).loc[:CUTOFF]
p=p.loc[:p.dropna(how='all').index.max()]
r=p.pct_change(); r60=p.pct_change(60); vol20=r.rolling(20,min_periods=15).std()
# Negative 60d drawdown magnitude continuously amplifies a weak-return/high-volatility reversal score.
dd=(p/p.rolling(60,min_periods=45).max()-1).clip(upper=0)
dd_weight=(-dd).div((-dd).mean(axis=1).replace(0,np.nan),axis=0).clip(0,3)
base=vol20.rank(axis=1,pct=True)-r60.rank(axis=1,pct=True)
f=(base*dd_weight).replace([np.inf,-np.inf],np.nan)
print('FACTOR continuous_drawdown_weighted_volatility_adjusted_medium_term_reversal_60_20d VALIDATED_THROUGH',p.index.max().date())
print('definition=(rank(20d daily-return volatility)-rank(60d return))*normalized magnitude of negative 60d drawdown, capped at 3')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in (1,5,10,20):
 y=p.shift(-h)/p-1; rec=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   x=spearmanr(q.f,q.y).statistic
   if np.isfinite(x): rec.append((d,x)); ns.append(len(q))
 s=pd.Series(dict(rec)); ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.4f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for n,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2034','2031-01-01',CUTOFF)]:
 s=ics[10].loc[lo:hi]; print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(n,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); v=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:v.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(v),len(v)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn));
  if z.get('validation',{}).get('status')=='EFFECTIVE':eff.append(z['factor_id'])
 except Exception: pass
scores=[]; missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits: missing.append(fid); continue
 old=pd.read_pickle(max(hits,key=os.path.getmtime)); q=pd.concat([f.stack().rename('candidate'),old.stack().rename('library')],axis=1).dropna()
 if len(q)<8 or q.candidate.nunique()<2 or q.library.nunique()<2: missing.append(fid); continue
 scores.append((abs(spearmanr(q.candidate,q.library).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
if len(scores)==len(eff): print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else: print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
f.to_pickle('scripts/miner_1_20340525_continuous_drawdown_weighted_volatility_adjusted_medium_term_reversal_60_20d_signal.pkl')

"""One idea: medium-term trend deceleration reversal, evaluated visibility-safe through prior completed day."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ASOF=pd.Timestamp('2034-06-21')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:load(a) for a in A}).loc[:ASOF]
p=p.loc[:p.dropna(how='all').index.max()]; r=p.pct_change()
# A recent acceleration relative to the preceding medium trend is expected to mean-revert.
# Positive score = deceleration/downshift: -(20-session return - one-third 60-session return).
f=-(p.pct_change(20)-p.pct_change(60)/3).replace([np.inf,-np.inf],np.nan)
print('FACTOR medium_term_trend_deceleration_reversal_20_60 VALIDATED_THROUGH',p.index.max().date())
print('definition=-(return_20 - return_60/3); high score identifies recent underperformance versus its 60-session trend')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in (1,5,10,20):
 y=p.shift(-h)/p-1; rec=[];ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z):rec.append((d,z));ns.append(len(q))
 s=pd.Series(dict(rec));ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.4f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for name,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2034','2031-01-01',p.index.max())]:
 s=ics[10].loc[lo:hi]
 print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(name,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: to.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(to),len(to)))
# Independence evidence is valid only for exact ID-matched historical matrices.
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn))
  if z.get('validation',{}).get('status')=='EFFECTIVE': eff.append(z['factor_id'])
 except Exception:pass
scores=[];missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits: missing.append(fid);continue
 old=pd.read_pickle(max(hits,key=os.path.getmtime))
 if not isinstance(old,pd.DataFrame): missing.append(fid);continue
 q=pd.concat([f.stack().rename('x'),old.stack().rename('y')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.y.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.x,q.y).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d exact_evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
if len(scores)==len(eff): print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else: print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE; missing_ids='+','.join(missing))
f.to_pickle('scripts/miner_1_20340622_medium_term_trend_deceleration_reversal_20_60_signal.pkl')

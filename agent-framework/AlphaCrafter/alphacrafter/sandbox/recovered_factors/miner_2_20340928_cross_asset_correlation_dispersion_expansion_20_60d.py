"""One idea: cross-asset correlation dispersion expansion, 20d relative to 60d."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2034-09-27')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:load(a) for a in A}).loc[:CUT]
p=p.reindex(pd.date_range(p.index.min(),p.index.max(),freq='B')).ffill()
r=p.pct_change()
# For each asset, dispersion of correlations to the other 14 instruments.  High
# values identify a recently more differentiated relationship structure rather
# than uniformly coupled exposure; use change versus a slow baseline.
d20={}; d60={}
for a in A:
 others=[b for b in A if b!=a]
 q20=pd.concat([r[a].rolling(20,min_periods=15).corr(r[b]) for b in others],axis=1)
 q60=pd.concat([r[a].rolling(60,min_periods=45).corr(r[b]) for b in others],axis=1)
 d20[a]=q20.std(axis=1,ddof=0); d60[a]=q60.std(axis=1,ddof=0)
f=pd.DataFrame(d20)-pd.DataFrame(d60)
cut=p.dropna(how='all').index.max()
print('FACTOR cross_asset_correlation_dispersion_expansion_20_60d VALIDATED_THROUGH',cut.date())
print('definition=std_j(corr20(return_i,return_j))-std_j(corr60(return_i,return_j)), j != i; high means recently more dispersed pairwise relationships')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 out=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):out.append((d,v));ns.append(len(q))
 s=pd.Series(dict(out),dtype=float); ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.3f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2033','2027-01-01','2033-12-31'),('2034_YTD','2034-01-01',cut)]:
 s=ics[5].loc[lo:hi]
 print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turn),len(turn)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn))
  if z.get('validation',{}).get('status')=='EFFECTIVE': eff.append(z['factor_id'])
 except Exception: pass
scores=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits: continue
 old=pd.read_pickle(max(hits,key=os.path.getmtime)); q=pd.concat([f.stack().rename('x'),old.stack().rename('z')],axis=1).dropna()
 if len(q)>=8 and q.x.nunique()>1 and q.z.nunique()>1: scores.append(abs(spearmanr(q.x,q.z).statistic))
print('INDEPENDENCE effective=%d evidence=%d max_abs_library_correlation=%s'%(len(eff),len(scores),('%.6f'%max(scores) if len(scores)==len(eff) and scores else 'UNAVAILABLE')))
f.to_pickle('scripts/miner_2_20340928_cross_asset_correlation_dispersion_expansion_20_60d_signal.pkl')

"""One candidate: residual intraday-range compression transition (20d vs 60d), visible through 2033-11-09."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-11-09')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
 D[a]=d
p=pd.DataFrame({a:D[a]['close'] for a in A}); r=p.pct_change()
# Daily range is normalized by close. Remove each day's cross-asset median to capture idiosyncratic liquidity/uncertainty state.
rng=pd.DataFrame({a:(D[a]['high']-D[a]['low']).abs()/D[a]['close'].replace(0,np.nan) for a in A})
res=rng.sub(rng.median(axis=1),axis=0)
# Negative signal means a recent idiosyncratic range compression relative to its baseline.
f=(res.rolling(20,min_periods=20).mean()/res.rolling(20,min_periods=20).std(ddof=1) - res.rolling(60,min_periods=60).mean()/res.rolling(60,min_periods=60).std(ddof=1)).replace([np.inf,-np.inf],np.nan)
print('FACTOR residual_intraday_range_compression_transition_20_60d VALIDATED_THROUGH',CUT.date())
print('definition=zmean20(idiosyncratic_range)-zmean60(idiosyncratic_range), idiosyncratic_range=(high-low)/close-cross_section_median; sign tests both orientations')
print('coverage=%.6f valid_dates=%d valid_cells=%d assets=%d'%(f.notna().mean().mean(),f.notna().any(axis=1).sum(),f.notna().sum().sum(),len(A)))
allres={}
for sign,label in [(1,'RAW'),(-1,'INVERSE')]:
 print('ORIENTATION',label)
 for h in [1,5,10,20]:
  vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
  for d in f.index:
   q=pd.concat([(sign*f.loc[d]).rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1:
    z=spearmanr(q.f,q.y).statistic
    if np.isfinite(z): vals.append((d,z)); ns.append(len(q))
  s=pd.Series(dict(vals),dtype=float); sd=s.std(ddof=1); ir=s.mean()/sd if sd else np.nan
  allres[(sign,h)]=s
  print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),ir,len(s),(s>0).mean(),np.mean(ns)))
  if h==10:
   for name,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01',CUT)]:
    x=s.loc[lo:hi]; print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(name,len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
# Turnover based on prospective orientation-independent ranks.
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turn),len(turn)))
# Contract audit: require artifacts for every effective library definition.
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  x=json.load(open(fn))
  if x.get('validation',{}).get('status')=='EFFECTIVE': eff.append(x['factor_id'])
 except Exception: pass
scores=[]; missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits: missing.append(fid); continue
 z=pd.read_pickle(max(hits,key=os.path.getmtime)); q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2: missing.append(fid); continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d comparable=%d missing=%d complete=%s'%(len(eff),len(scores),len(missing),not missing))
if scores: print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
print('ADMISSION=FAIL unless one same-horizon orientation meets gates AND complete independence evidence has max_abs_rho < 0.5000')
f.to_pickle('scripts/miner_1_20331110_residual_intraday_range_compression_transition_20_60d_signal.pkl')

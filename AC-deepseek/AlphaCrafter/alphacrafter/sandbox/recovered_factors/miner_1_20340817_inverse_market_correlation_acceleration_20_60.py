"""One idea: inverse residual market-correlation acceleration (20/60d)."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2034-08-16')
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
p=p.loc[:p.dropna(how='all').index.max()]; r=p.pct_change(fill_method=None)
# Higher score means its 20d linkage to the contemporaneous cross-asset median has declined versus 60d linkage.
market=r.median(axis=1)
c20=pd.DataFrame({a:r[a].rolling(20,min_periods=16).corr(market) for a in A})
c60=pd.DataFrame({a:r[a].rolling(60,min_periods=48).corr(market) for a in A})
f=(-(c20-c60)).replace([np.inf,-np.inf],np.nan)
print('FACTOR inverse_market_correlation_acceleration_20_60 VALIDATED_THROUGH',p.index.max().date())
print('expression=-(rolling_corr(return_i,cross_asset_median_return,20)-rolling_corr(return_i,cross_asset_median_return,60))')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
stats={}
for h in (1,5,10,20):
 y=p.shift(-h).div(p)-1; rows=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):rows.append((d,v));ns.append(len(q))
 s=pd.Series(dict(rows),dtype=float);stats[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.4f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2034','2031-01-01',p.index.max())]:
 s=stats[10].loc[lo:hi];print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True);tr=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:tr.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(tr),len(tr)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  x=json.load(open(fn));
  if x.get('validation',{}).get('status')=='EFFECTIVE':eff.append(x['factor_id'])
 except:pass
scores=[];missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits:missing.append(fid);continue
 z=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('x'),z.stack().rename('y')],axis=1).dropna()
 if len(q)<8 or q.x.nunique()<2 or q.y.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.x,q.y).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d exact_evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
if len(scores)==len(eff):print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else:print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE; missing_ids='+','.join(missing))
f.to_pickle('scripts/miner_1_20340817_inverse_market_correlation_acceleration_20_60_signal.pkl')

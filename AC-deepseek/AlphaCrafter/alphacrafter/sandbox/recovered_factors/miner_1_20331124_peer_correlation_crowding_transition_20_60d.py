"""One candidate: peer-correlation diversification transition (20d versus 60d), validated through prior completed day."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-11-23')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change()
def peer_corr(w):
 out=pd.DataFrame(np.nan,index=r.index,columns=A)
 for i in range(w-1,len(r)):
  c=r.iloc[i-w+1:i+1].corr(method='pearson')
  # Equal-weighted linkage to other available assets; excludes own diagonal.
  out.iloc[i]=[(c.loc[a].drop(a).mean() if a in c else np.nan) for a in A]
 return out
# Positive when an asset has become more correlated/crowded than its own baseline.
f=(peer_corr(20)-peer_corr(60)).replace([np.inf,-np.inf],np.nan)
print('FACTOR peer_correlation_crowding_transition_20_60d VALIDATED_THROUGH',CUT.date())
print('expression=mean_peer_corr_20d(asset)-mean_peer_corr_60d(asset)')
print('coverage=%.6f calendar_dates=%d valid_cells=%d assets=%d'%(f.notna().mean().mean(),f.notna().any(axis=1).sum(),f.notna().sum().sum(),len(A)))
all_stats={}
for orient,ff in [('raw',f),('inverse',-f)]:
 for h in [1,5,10,20]:
  vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
  for d in ff.index:
   q=pd.concat([ff.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
    z=spearmanr(q.f,q.y).statistic
    if np.isfinite(z): vals.append((d,z)); ns.append(len(q))
  s=pd.Series(dict(vals),dtype=float); sd=s.std(ddof=1); ir=s.mean()/sd if sd and np.isfinite(sd) else np.nan
  all_stats[(orient,h)]=(s,ir)
  print('%s H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(orient,h,s.mean(),ir,len(s),(s>0).mean(),np.mean(ns)))
  if h==10:
   for name,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01',CUT)]:
    x=s.loc[lo:hi]; print('REGIME10 %s %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(orient,name,len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover_raw=%.6f pairs=%d'%(np.mean(turn),len(turn)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  x=json.load(open(fn))
  if x.get('validation',{}).get('status')=='EFFECTIVE': eff.append(x['factor_id'])
 except Exception: pass
found=[]; scores=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if hits:
  found.append(fid); z=pd.read_pickle(max(hits,key=os.path.getmtime)); q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1: scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE artifacts=%d effective=%d missing=%d'%(len(found),len(eff),len(eff)-len(found)))
if scores: print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else: print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
print('ADMISSION requires complete artifact evidence and max correlation <0.5000.')
f.to_pickle('scripts/miner_1_20331124_peer_correlation_crowding_transition_20_60d_signal.pkl')

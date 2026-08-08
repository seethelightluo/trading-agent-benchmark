"""One candidate: peer-correlation crowding transition (20d minus 60d), validated through completed 2033-09-28."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-09-28')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change()
# For each asset, average contemporaneous correlation with the other 14 assets.
# Positive values denote recently increased cross-asset crowding relative to baseline.
def peer_corr(window):
 out=pd.DataFrame(index=r.index,columns=A,dtype=float)
 for end in range(window-1,len(r)):
  c=r.iloc[end-window+1:end+1].corr()
  out.iloc[end]=[(c.loc[a].drop(a).mean()) for a in A]
 return out
f=(peer_corr(20)-peer_corr(60)).replace([np.inf,-np.inf],np.nan)
print('FACTOR peer_correlation_crowding_transition_20_60d VALIDATED_THROUGH',CUT.date())
print('coverage=%.6f valid_dates=%d valid_cells=%d assets=%d'%(f.notna().mean().mean(),f.notna().any(axis=1).sum(),f.notna().sum().sum(),len(A)))
for h in [1,5,10,20]:
 out=[];ns=[];fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z):out.append((d,z));ns.append(len(q))
 s=pd.Series(dict(out),dtype=float); sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for name,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01',CUT)]:
   x=s.loc[lo:hi];print('REGIME10',name,'dates',len(x),'IC=%+.6f ICIR=%+.6f hit=%.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:to.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(to),len(to)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  x=json.load(open(fn))
  if x.get('validation',{}).get('status')=='EFFECTIVE':eff.append(x['factor_id'])
 except Exception: pass
found=[];scores=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits:continue
 found.append(fid); z=pd.read_pickle(max(hits,key=os.path.getmtime))
 q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1:scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE artifacts=%d effective=%d'%(len(found),len(eff)))
if scores:print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
print('ADMISSION=FAIL if artifacts != effective: missing correlations are failed evidence.')
f.to_pickle('scripts/miner_3_20330929_peer_correlation_crowding_transition_20_60d_signal.pkl')

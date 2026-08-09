"""One candidate: transition in each asset's linkage to defensive-minus-growth cross-asset leadership, through completed bar 2033-11-23."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-11-23')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change()
# Observable cross-asset leadership spread: defensive precious metal and rates versus global technology/crypto risk growth.
defensive=r[['XAU','US10Y','CN10Y']].mean(axis=1)
growth=r[['NDX','SOX','000688.SH','BTC','ETH']].mean(axis=1)
lead=defensive-growth
def beta(w):
 out=pd.DataFrame(np.nan,index=r.index,columns=A)
 for i in range(w-1,len(r)):
  rr=r.iloc[i-w+1:i+1]; x=lead.iloc[i-w+1:i+1]; ok=x.notna(); xx=x[ok]; yy=rr.loc[ok]; den=xx.var()
  if np.isfinite(den) and den>0: out.iloc[i]=yy.sub(yy.mean()).mul(xx-xx.mean(),axis=0).mean()/den
 return out
# Positive signals a rising recent loading to defensive-over-growth leadership.
f=(beta(20)-beta(60)).replace([np.inf,-np.inf],np.nan)
print('FACTOR defensive_growth_leadership_loading_transition_20_60d VALIDATED_THROUGH',CUT.date())
print('definition=beta20(asset return, XAU/US10Y/CN10Y mean return minus NDX/SOX/000688/BTC/ETH mean return)-beta60')
print('coverage=%.6f valid_dates=%d valid_cells=%d assets=%d'%(f.notna().mean().mean(),f.notna().any(axis=1).sum(),f.notna().sum().sum(),len(A)))
for h in [1,5,10,20]:
 vals=[];ns=[];fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):vals.append((d,v));ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float); ir=s.mean()/s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),ir,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01',CUT)]:
   z=s.loc[lo:hi];print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(n,len(z),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:to.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(to),len(to)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn));
  if z.get('validation',{}).get('status')=='EFFECTIVE':eff.append(z['factor_id'])
 except:pass
found=[];scores=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if hits:
  found.append(fid);z=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1:scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE artifacts=%d effective=%d'%(len(found),len(eff)))
if scores:print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
print('ADMISSION=FAIL if artifacts != effective; missing complete correlation evidence fails contract.')
f.to_pickle('scripts/miner_3_20331124_defensive_growth_leadership_loading_transition_20_60d_signal.pkl')

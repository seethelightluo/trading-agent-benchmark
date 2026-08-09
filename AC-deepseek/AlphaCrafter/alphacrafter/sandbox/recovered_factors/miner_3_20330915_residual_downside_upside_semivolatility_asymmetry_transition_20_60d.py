"""One candidate: residual downside/upside semivolatility asymmetry transition, 20d versus 60d.
Completed bars only through 2033-09-14; missing library signals are explicitly a failed audit."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-09-14')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change(); market=r.mean(axis=1)
# Each asset is residualized using a trailing 60-session market beta, so the feature
# describes asymmetric idiosyncratic risk rather than simply market downside risk.
beta=pd.DataFrame({a:r[a].rolling(60,min_periods=42).cov(market)/(market.rolling(60,min_periods=42).var()+1e-12) for a in A})
e=r-beta.mul(market,axis=0)
def asym(w,mp):
 neg=e.where(e<0).pow(2).rolling(w,min_periods=mp).mean().pow(.5)
 pos=e.where(e>0).pow(2).rolling(w,min_periods=mp).mean().pow(.5)
 return np.log((neg+1e-10)/(pos+1e-10))
f=(asym(20,14)-asym(60,42)).replace([np.inf,-np.inf],np.nan)
print('FACTOR residual_downside_upside_semivolatility_asymmetry_transition_20_60d VALIDATED_THROUGH',CUT.date())
print('coverage=%.6f valid_dates=%d valid_cells=%d assets=%d'%(f.notna().mean().mean(),f.notna().any(axis=1).sum(),f.notna().sum().sum(),len(A)))
res={}
for h in [1,5,10,20]:
 out=[];ns=[];fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):out.append((d,v));ns.append(len(q))
 s=pd.Series(dict(out),dtype=float);res[h]=s; sd=s.std(ddof=1)
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(ns)))
 if h==10:
  for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01',CUT)]:
   x=s.loc[lo:hi]; print('REGIME10',n,'dates',len(x),'IC=%+.6f ICIR=%+.6f hit=%.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:tos.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(tos),len(tos)))
# Audit against available persisted signal artifacts. Contract requires all 30;
# this cannot be converted to a passing zero-correlation assumption.
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  x=json.load(open(fn))
  if x.get('validation',{}).get('status')=='EFFECTIVE':eff.append(x['factor_id'])
 except: pass
scores=[]; found=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits:continue
 found.append(fid); x=pd.read_pickle(max(hits,key=os.path.getmtime))
 q=pd.concat([f.stack().rename('a'),x.stack().rename('b')],axis=1).dropna()
 if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1:scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE artifacts=%d effective=%d'% (len(found),len(eff)))
if scores: print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
print('ADMISSION=FAIL if artifacts != effective: complete all-admitted definition reconstruction is required.')
f.to_pickle('scripts/miner_3_20330915_residual_downside_upside_semivolatility_asymmetry_transition_20_60d_signal.pkl')

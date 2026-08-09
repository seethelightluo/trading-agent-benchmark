"""Miner 2 scheduled revalidation: commonality expansion transition 40, all admitted-library novelty audit."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; C={}; VV={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date'); C[a]=pd.to_numeric(d.close,errors='coerce'); VV[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C).sort_index(); V=pd.DataFrame(VV).reindex(P.index); r=P.pct_change(fill_method=None); m=r.median(axis=1); cutoff=P.dropna(how='all').index.max(); H=[1,5,10,20]
other=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A}); corr20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(other[a]) for a in A})
F=corr20.rolling(20,min_periods=15).mean()-corr20.shift(20).rolling(20,min_periods=15).mean(); F=F.sub(F.median(axis=1),axis=0)
fw={h:P.shift(-h)/P-1 for h in H}
def ev(h,span=None):
 x=F if span is None else F.loc[span[0]:span[1]]; y=fw[h].reindex(x.index); z=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v); ns.append(len(q))
 z=np.array(z); sd=z.std(ddof=1) if len(z)>1 else np.nan
 return dict(dates=len(z),ic=round(float(z.mean()),6),icir=round(float(z.mean()/sd),6),hit=round(float((z>0).mean()),6),mean_n=round(float(np.mean(ns)),3),min_n=int(min(ns)))
print('FACTOR commonality_expansion_transition_40 CUTOFF',cutoff.date(),'PERIOD',P.index.min().date(),cutoff.date(),'ASSETS',len(A)); print('CELLS',int(F.notna().sum().sum()),'/',F.size,'COVERAGE',round(float(F.notna().stack().mean()),6))
for h in H: print('HORIZON',h,ev(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,ev(10,s))
print('TURNOVER_RANK_CHANGE',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
# Reuse maintained complete admitted-library reconstruction, then remove self before correlation audit.
code=open('scripts/miner_3_20280601_dispersion_shock_beta_resilience_40.py').read(); section=code.split('# Full admitted-library signal reconstruction.')[1].split("mx=0;who='';cells=0")[0]; exec(section)
S.pop('commonexpand',None); mx=0.; who=''; cells=0; evidence=0
for n,g in S.items():
 q=pd.concat([F.stack(),g.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>2 else np.nan
 if np.isfinite(rho): evidence+=1
 print('LIBCORR',n,'CELLS',len(q),'RHO',round(float(rho),6) if np.isfinite(rho) else None)
 if np.isfinite(rho) and abs(rho)>mx: mx=float(abs(rho));who=n;cells=len(q)
print('LIBRARY_EVIDENCE',evidence,'OF',len(S)); print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',who,'EVIDENCE_CELLS',cells,'LIBRARY_FACTORS',len(S))

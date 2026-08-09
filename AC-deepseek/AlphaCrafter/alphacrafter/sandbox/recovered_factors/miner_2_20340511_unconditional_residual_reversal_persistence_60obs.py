"""Validate dense unconditional residual reversal persistence, with full admitted-library audit."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
# Reuse completed-bar data preparation and exact admitted-library reconstruction.
src=open('scripts/miner_3_20320624_revalidate_common_stress_repair_rank_migration_60obs.py').read()
src=src.replace("END=pd.Timestamp('2032-06-23')", "END=pd.Timestamp('2034-05-10')")
exec(src.split("print('REVALIDATION common_stress")[0])
# At decision t, residuals are r_i minus equal-weight market r; signal uses t-1
# and earlier only.  Higher score denotes greater historical residual reversal.
f=-res.shift(1).rolling(60,min_periods=45).corr(res.shift(1).shift(1))
print('CANDIDATE unconditional_residual_reversal_persistence_60obs endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H); z=[]; ds=[]; ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ds.append(t); ns.append(len(q))
 z=np.asarray(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
best=max(R,key=lambda h:abs(R[h][0].mean())*abs(R[h][0].mean()/R[h][0].std(ddof=1)));z,ds,_=R[best];print('SELECTED',best)
for nm,st,en in [('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('recent_2033_34','2033-01-01','2034-05-10')]:
 q=z[(ds>=st)&(ds<=en)];print('REGIME',nm,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None,'hit',round((q>0).mean(),6) if len(q) else None)
rnk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8:turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('QUALITY turnover',round(np.mean(turns),6),'comparisons',len(turns),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
# Audit against every signal reconstructed by the current-library audit source.
audit=src[src.index('# Full current-library reconstruction'):];audit=audit[:audit.index('mx=-1')];exec(audit)
hi_state=disp.shift(1).gt(disp.rolling(60,min_periods=45).median().shift(1)).astype(float);den=hi_state.rolling(60,min_periods=45).sum()
L['dispersion_conditioned_residual_resilience_60obs']=res.shift(1).rank(axis=1,pct=True).mul(hi_state,axis=0).rolling(60,min_periods=45).sum().div(den.replace(0,np.nan),axis=0).where(den>=12)
resolution=(disp.shift(1)>disp.rolling(60,min_periods=45).median().shift(1))&(disp<disp.rolling(20,min_periods=15).median())
L['dispersion_resolution_residual_rebound_60obs']=res.shift(1).rank(axis=1,pct=True).where(resolution).rolling(60,min_periods=45).mean()
mx=-1;who=None;cells=0;missing=[]
for n,s in L.items():
 q=pd.concat([f.stack(),s.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if not len(q):missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)

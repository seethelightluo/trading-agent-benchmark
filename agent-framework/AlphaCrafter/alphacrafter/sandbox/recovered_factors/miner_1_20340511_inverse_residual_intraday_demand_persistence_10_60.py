"""Completed-bar validation: inverse residual intraday demand persistence, 10/60."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
# Stable base loads panels and exact maintained library definitions.
src=open('scripts/miner_3_20320624_revalidate_common_stress_repair_rank_migration_60obs.py').read()
src=src.replace("END=pd.Timestamp('2032-06-23')", "END=pd.Timestamp('2034-05-10')")
exec(src.split("print('REVALIDATION common_stress")[0])
op=pd.DataFrame({a:ld(a,'open').replace(0,np.nan) for a in A}).reindex(p.index)
intra=p.div(op)-1; common=intra.median(axis=1)
bintra=pd.DataFrame({a:intra[a].rolling(60,min_periods=45).cov(common)/common.rolling(60,min_periods=45).var() for a in A})
f=-(intra-bintra.mul(common,axis=0)).rolling(10,min_periods=8).mean()
print('CANDIDATE inverse_residual_intraday_demand_persistence_10_60obs endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H); z=[];ds=[];ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'mean_names',round(np.mean(ns),3),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
for H,(z,ds,_) in R.items():
 for nm,st,en in [('2020_2027','2020-01-01','2027-12-31'),('2028_2030','2028-01-01','2030-12-31'),('2031_now','2031-01-01','2034-05-10'),('latest_6m','2033-11-10','2034-05-10')]:
  x=z[(ds>=st)&(ds<=en)];print('REGIME H',H,nm,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rnk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8: turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('QUALITY turnover',round(np.mean(turn),6),'comparisons',len(turn),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
# Reconstruct all maintained definitions, then append later admitted ones.
audit=src[src.index('# Full current-library reconstruction'):]; audit=audit[:audit.index('mx=-1')];exec(audit)
disp=r.std(axis=1);hi_state=disp.shift(1).gt(disp.rolling(60,min_periods=45).median().shift(1)).astype(float);den=hi_state.rolling(60,min_periods=45).sum()
L['dispersion_conditioned_residual_resilience_60obs']=res.shift(1).rank(axis=1,pct=True).mul(hi_state,axis=0).rolling(60,min_periods=45).sum().div(den.replace(0,np.nan),axis=0).where(den>=12)
resolution=(disp.shift(1)>disp.rolling(60,min_periods=45).median().shift(1))&(disp<disp.rolling(20,min_periods=15).median())
L['dispersion_resolution_residual_rebound_60obs']=res.shift(1).rank(axis=1,pct=True).where(resolution).rolling(60,min_periods=45).mean()
mx=-1;who=None;cells=0;missing=[]
for n,s in L.items():
 q=pd.concat([f.stack(),s.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)<8:missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if not np.isfinite(rho):missing.append(n);continue
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'closest',who,'evidence_cells',cells,'signals_tested',len(L),'missing',missing)

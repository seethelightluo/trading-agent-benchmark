"""Miner_2: VIX shock-conditioned idiosyncratic response, 60 observations."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
src=open('scripts/miner_3_20320624_revalidate_common_stress_repair_rank_migration_60obs.py').read()
src=src.replace("END=pd.Timestamp('2032-06-23')", "END=pd.Timestamp('2034-08-30')")
exec(src.split("print('REVALIDATION common_stress")[0])
# At t, estimate each asset's average residual return on sessions immediately after
# an unusually large VIX move, using strictly lagged 60 completed observations.
vix=macro('VIX')
shock=(vix.abs().shift(1)>=vix.abs().shift(1).rolling(60,min_periods=30).quantile(.80))
res=r.shift(1).sub(beta(r.shift(1),m.shift(1),60)*m.shift(1),axis=0)
f=pd.DataFrame(np.nan,index=r.index,columns=A)
for t in range(60,len(r)):
 ix=r.index[t-60:t]; mask=shock.loc[ix]
 for a in A:
  z=res.loc[ix,a][mask].dropna()
  if len(z)>=8:f.loc[r.index[t],a]=z.mean()
print('FACTOR vix_shock_conditioned_idiosyncratic_response_60obs endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H); z=[];ds=[];ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
best=max(R,key=lambda h:abs(R[h][0].mean())*abs(R[h][0].mean()/R[h][0].std(ddof=1)));z,ds,_=R[best];print('SELECTED',best)
for nm,st,en in [('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('recent_2033_34','2033-01-01','2034-08-30')]:
 q=z[(ds>=st)&(ds<=en)];print('REGIME',nm,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),6))
rnk=f.rank(axis=1,pct=True);ch=(rnk-rnk.shift()).abs().stack();print('QUALITY turnover',round(ch.mean(),6),'comparisons',len(ch),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
audit=src[src.index('# Full current-library reconstruction'):];audit=audit[:audit.index('mx=-1')];exec(audit)
mx=-1;who=None;cells=0;missing=[]
for n,s in L.items():
 q=pd.concat([f.stack(),s.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if not len(q):missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)
f.to_pickle('scripts/miner_2_20340831_vix_shock_conditioned_idiosyncratic_response_60obs_signal.pkl')

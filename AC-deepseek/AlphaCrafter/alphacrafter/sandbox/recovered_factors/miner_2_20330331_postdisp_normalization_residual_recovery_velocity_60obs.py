"""Validate post-dispersion-normalization residual recovery velocity, completed-bar only."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
src=open('scripts/miner_3_20320624_revalidate_common_stress_repair_rank_migration_60obs.py').read()
src=src.replace("END=pd.Timestamp('2032-06-23')", "END=pd.Timestamp('2033-03-30')")
exec(src.split("print('REVALIDATION common_stress")[0])
# A resolution state: dispersion was elevated two completed sessions ago and is
# below its short-run median on the most recently completed session.  At each
# such transition, score the asset's completed five-session idiosyncratic return.
disp=r.std(axis=1)
high_two_ago=disp.shift(2).gt(disp.rolling(60,min_periods=45).median().shift(2))
normalized=disp.shift(1).lt(disp.rolling(20,min_periods=15).median().shift(1))
resolution=(high_two_ago & normalized)
recovery=res.shift(1).rolling(5,min_periods=4).sum()
count=resolution.astype(float).rolling(60,min_periods=45).sum()
f=recovery.where(resolution,axis=0).rolling(60,min_periods=45).mean().where(count>=8)
print('CANDIDATE post_dispersion_normalization_residual_recovery_velocity_60obs endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6),'resolution_sessions',int(resolution.sum()))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H); z=[]; ds=[]; ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.asarray(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
z,ds,_=R[20]
for name,a,bnd in [('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('recent_2033','2033-01-01','2033-03-30')]:
 x=z[(ds>=a)&(ds<=bnd)]; print('REGIME',name,'dates',len(x),'IC',round(x.mean(),6) if len(x) else None,'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None,'hit',round((x>0).mean(),6) if len(x) else None)
rnk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('QUALITY turnover',round(np.mean(turn),6),'comparisons',len(turn),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
# Reconstruct the standard 30 signals plus both newer dispersion factors.
audit=src[src.index('# Full current-library reconstruction'):];audit=audit[:audit.index('mx=-1')];exec(audit)
hi=(disp.shift(1)>disp.rolling(60,min_periods=45).median().shift(1)).astype(float)
den=hi.rolling(60,min_periods=45).sum()
L['dispersion_conditioned_residual_resilience_60obs']=res.shift(1).rank(axis=1,pct=True).mul(hi,axis=0).rolling(60,min_periods=45).sum().div(den.replace(0,np.nan),axis=0).where(den>=12)
old_resolution=(disp.shift(1)>disp.rolling(60,min_periods=45).median().shift(1)) & (disp<disp.rolling(20,min_periods=15).median())
L['dispersion_resolution_residual_rebound_60obs']=res.shift(1).rank(axis=1,pct=True).where(old_resolution).rolling(60,min_periods=45).mean()
mx=-1;missing=[]
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if not len(q): missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)

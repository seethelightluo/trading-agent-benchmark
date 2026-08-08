"""Exact revalidation of admitted dispersion-resolution residual rebound, completed bars."""
import pandas as pd,numpy as np
from scipy.stats import spearmanr
src=open('scripts/miner_3_20320624_revalidate_common_stress_repair_rank_migration_60obs.py').read().replace("END=pd.Timestamp('2032-06-23')","END=pd.Timestamp('2033-04-27')")
exec(src.split("print('REVALIDATION common_stress")[0])
# Exact persisted definition: resolution ending t-1 (elevated dispersion at t-2 then decline),
# rank residual return cumulatively from t-5..t-1, average event ranks in a 60-session history.
disp=r.std(axis=1); base=disp.rolling(60,min_periods=45).median()
event=(disp.shift(2)>base.shift(2)) & (disp.shift(1)<disp.shift(2))
rebound=res.shift(1).rolling(5,min_periods=5).sum().rank(axis=1,pct=True)
count=event.astype(float).rolling(60,min_periods=45).sum()
f=rebound.where(event,axis=0).rolling(60,min_periods=45).mean().where(count>=8)
print('REVALIDATION dispersion_resolution_residual_rebound_60obs endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6),'events',int(event.sum()))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H); z=[]; ds=[]; ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
z,ds,_=R[20]
for nm,a,bx in [('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('recent_2033','2033-01-01','2033-04-27')]:
 x=z[(ds>=a)&(ds<=bx)];print('REGIME',nm,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rnk=f.rank(axis=1,pct=True); tt=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8:tt.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('QUALITY turnover',round(np.mean(tt),6),'comparisons',len(tt),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
# Base full-library signals plus exact other admitted Miner-2 dispersion signal.
audit=src[src.index('# Full current-library reconstruction'):];audit=audit[:audit.index('mx=-1')];exec(audit)
high=(disp.shift(1)>base.shift(1)).astype(float); den=high.rolling(60,min_periods=45).sum()
L['dispersion_conditioned_residual_resilience_60obs']=res.shift(1).rank(axis=1,pct=True).mul(high,axis=0).rolling(60,min_periods=45).sum().div(den.replace(0,np.nan),axis=0).where(den>=12)
mx=-1;miss=[]
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if not len(q):miss.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',miss)

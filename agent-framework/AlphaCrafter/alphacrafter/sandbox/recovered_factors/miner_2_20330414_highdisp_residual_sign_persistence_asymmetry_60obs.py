"""Validate high-dispersion residual sign-persistence asymmetry (60 sessions), completed-bar only."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
src=open('scripts/miner_3_20320624_revalidate_common_stress_repair_rank_migration_60obs.py').read()
src=src.replace("END=pd.Timestamp('2032-06-23')", "END=pd.Timestamp('2033-04-13')")
exec(src.split("print('REVALIDATION common_stress")[0])
# At high prior cross-asset dispersion, compare how often an asset's residual
# negative/positive sign persists from the prior completed session.  This is
# event persistence, not residual magnitude or recovery level.
disp=r.std(axis=1)
high=disp.shift(1).gt(disp.rolling(60,min_periods=45).median().shift(1))
e=res.shift(1)
eprev=res.shift(2)
neg=(high & e.lt(0) & eprev.lt(0)).astype(float)
pos=(high & e.gt(0) & eprev.gt(0)).astype(float)
negbase=(high & e.lt(0)).astype(float); posbase=(high & e.gt(0)).astype(float)
# Smoothed conditional persistence probabilities over the trailing 60 sessions.
nn=neg.rolling(60,min_periods=45).sum(); np_=pos.rolling(60,min_periods=45).sum()
nd=negbase.rolling(60,min_periods=45).sum(); pd_=posbase.rolling(60,min_periods=45).sum()
f=(nn.add(0.5).div(nd.add(1.0))-np_.add(0.5).div(pd_.add(1.0))).where((nd>=8)&(pd_>=8))
print('CANDIDATE highdisp_residual_sign_persistence_asymmetry_60obs endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6),'high_sessions',int(high.sum()))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H); z=[];ds=[];ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'PASS',abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084)
z,ds,_=R[20]
for nm,a,b in [('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('recent_2033','2033-01-01','2033-04-13')]:
 x=z[(ds>=a)&(ds<=b)];print('REGIME',nm,'dates',len(x),'IC',round(x.mean(),6) if len(x) else None,'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None,'hit',round((x>0).mean(),6) if len(x) else None)
rnk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8:turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('QUALITY turnover',round(np.mean(turns),6),'comparisons',len(turns),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
# Exact reconstructed library, supplemented with the two recently admitted dispersion signals.
audit=src[src.index('# Full current-library reconstruction'):];audit=audit[:audit.index('mx=-1')];exec(audit)
hi=high.astype(float);den=hi.rolling(60,min_periods=45).sum()
L['dispersion_conditioned_residual_resilience_60obs']=res.shift(1).rank(axis=1,pct=True).mul(hi,axis=0).rolling(60,min_periods=45).sum().div(den.replace(0,np.nan),axis=0).where(den>=12)
oldres=(disp.shift(1)>disp.rolling(60,min_periods=45).median().shift(1)) & (disp<disp.rolling(20,min_periods=15).median())
L['dispersion_resolution_residual_rebound_60obs']=res.shift(1).rank(axis=1,pct=True).where(oldres).rolling(60,min_periods=45).mean()
mx=-1;missing=[];who=None;cells=0
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if not len(q):missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)

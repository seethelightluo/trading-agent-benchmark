"""Miner 2: normal-dispersion residual extreme-rank normalization, 60 observations."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
src=open('scripts/miner_3_20320624_revalidate_common_stress_repair_rank_migration_60obs.py').read().replace("END=pd.Timestamp('2032-06-23')","END=pd.Timestamp('2033-08-03')")
exec(src.split("print('REVALIDATION common_stress")[0])
# On quiet/normal dispersion sessions, measure whether an asset's idiosyncratic
# residual rank moves from an extreme toward the cross-sectional middle.  The
# factor is the negative change in distance-to-median following an extreme;
# larger values mean repeatable rank normalization rather than persistence.
disp=r.std(axis=1); normal=disp.shift(1).le(disp.rolling(60,min_periods=45).median().shift(1))
rk=res.rank(axis=1,pct=True)
d0=rk.shift(2).sub(.5).abs(); d1=rk.shift(1).sub(.5).abs()
event=normal & d0.max(axis=1).ge(.30) # timing only; individual qualification below
move=(d0-d1).where(d0.ge(.30)).where(normal,axis=0)
nobs=move.notna().astype(float).rolling(60,min_periods=20).sum()
f=move.rolling(60,min_periods=20).mean().where(nobs>=20)
print('CANDIDATE normal_dispersion_residual_extreme_rank_normalization_60obs endpoint',p.index.max().date(),'assets',len(A),'normal_days',int(normal.sum()),'eligible_dates',int(f.notna().any(axis=1).sum()),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
R={}
for H in (1,5,10,20):
 y=p.pct_change(H,fill_method=None).shift(-H); z=[];ds=[];ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'PASS',bool(abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084))
best=max(R,key=lambda h:abs(R[h][0].mean()*R[h][0].mean()/R[h][0].std(ddof=1)));z,ds,_=R[best];print('SELECTED_HORIZON',best)
for nm,a,b in [('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('recent_2033','2033-01-01','2033-08-03')]:
 x=z[(ds>=a)&(ds<=b)];print('REGIME',nm,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rnk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('QUALITY turnover',round(np.mean(turn),6),'comparisons',len(turn),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
audit=src[src.index('# Full current-library reconstruction'):];exec(audit[:audit.index('mx=-1')])
lag=res.shift(1).where(normal,axis=0);dn=(-lag.clip(upper=0)).rolling(60,min_periods=20).mean();up=lag.clip(lower=0).rolling(60,min_periods=20).mean()
L['normal_dispersion_residual_downside_upside_magnitude_asymmetry_60obs']=(dn-up).div(dn+up+1e-12).where(normal.astype(float).rolling(60,min_periods=45).sum()>=20,axis=0)
mx=-1;missing=[]
for n,x in L.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)<8:missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)

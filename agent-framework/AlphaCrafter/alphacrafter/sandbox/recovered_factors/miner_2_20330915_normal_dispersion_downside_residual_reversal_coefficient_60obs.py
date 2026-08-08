"""Miner 2: normal-dispersion downside-residual reversal coefficient, 60 sessions."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
src=open('scripts/miner_3_20320624_revalidate_common_stress_repair_rank_migration_60obs.py').read().replace("END=pd.Timestamp('2032-06-23')","END=pd.Timestamp('2033-09-14')")
exec(src.split("print('REVALIDATION common_stress")[0])
# Measure whether negative idiosyncratic shocks specifically reverse next session,
# avoiding the mixing of positive continuation with downside repair behavior.
disp=r.std(axis=1); high=disp.shift(1).gt(disp.rolling(60,min_periods=45).median().shift(1))
x=res.shift(2); y=res.shift(1)
valid=(~high).values[:,None] & x.lt(0).values
xx=x.where(valid); yy=y.where(valid)
count=xx.notna().astype(float).rolling(60,min_periods=20).sum()
f=(-xx.rolling(60,min_periods=20).corr(yy)).where(count>=20)
print('CANDIDATE normal_dispersion_downside_residual_reversal_coefficient_60obs endpoint',p.index.max().date(),'assets',len(A),'normal_days',int((~high).sum()),'eligible_dates',int(f.notna().any(axis=1).sum()),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
R={}
for H in (1,5,10,20):
 yfw=p.pct_change(H,fill_method=None).shift(-H); z=[];ds=[];ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],yfw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);ns.append(len(q))
 z=np.array(z);ds=pd.DatetimeIndex(ds);R[H]=(z,ds,ns);ir=z.mean()/z.std(ddof=1)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(ir,6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'PASS',bool(abs(z.mean())>=.007 and abs(ir)>=.084))
best=max(R,key=lambda h:abs(R[h][0].mean()*R[h][0].mean()/R[h][0].std(ddof=1));z,ds,_=R[best];print('SELECTED_HORIZON',best)
for nm,a,b in [('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('recent_2033','2033-01-01','2033-09-14')]:
 vv=z[(ds>=a)&(ds<=b)];print('REGIME',nm,'dates',len(vv),'IC',round(vv.mean(),6),'ICIR',round(vv.mean()/vv.std(ddof=1),6),'hit',round((vv>0).mean(),6))
rnk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('QUALITY turnover',round(np.mean(turn),6),'comparisons',len(turn),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
audit=src[src.index('# Full current-library reconstruction'):];exec(audit[:audit.index('mx=-1')])
# Include miner_2 factors introduced after the shared reconstruction.
lag=res.shift(1).where(~high,axis=0);dn=(-lag.clip(upper=0)).rolling(60,min_periods=20).mean();up=lag.clip(lower=0).rolling(60,min_periods=20).mean()
L['normal_dispersion_residual_downside_upside_magnitude_asymmetry_60obs']=(dn-up).div(dn+up+1e-12).where((~high).astype(float).rolling(60,min_periods=45).sum()>=20,axis=0)
nrk=res.rank(axis=1,pct=True);nd=(nrk.shift(2)-.5).abs();nmv=(nd-(nrk.shift(1)-.5).abs()).where(nd.ge(.30)).where(~high,axis=0);nn=nmv.notna().astype(float).rolling(60,min_periods=20).sum()
L['normal_dispersion_residual_extreme_rank_normalization_60obs']=nmv.rolling(60,min_periods=20).mean().where(nn>=20)
mx=-1;missing=[]
for n,v in L.items():
 q=pd.concat([f.stack(),v.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)<8:missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)

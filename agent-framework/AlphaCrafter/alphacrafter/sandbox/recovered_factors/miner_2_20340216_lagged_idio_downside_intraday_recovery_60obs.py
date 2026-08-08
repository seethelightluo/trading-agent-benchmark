"""Miner 2: lagged idiosyncratic downside-to-intraday recovery efficiency, 60 sessions."""
import pandas as pd, numpy as np
old=open('scripts/miner_2_20331222_vix_normalized_lagged_common_shock_response_60obs.py').read()
head=old[:old.index("# Asset-specific")].replace("END=pd.Timestamp('2033-12-21')", "END=pd.Timestamp('2034-02-15')")
exec(head)
# Price-only: after an idiosyncratic loss on the prior completed bar, measure how
# efficiently the asset recovers during the following session (open-to-close).
# Residualizing the trigger against the equal-weight universe avoids merely
# encoding broad market direction. All inputs at t use t-1 trigger and t bar.
iday=np.log(p.clip(lower=1e-12).div(op.clip(lower=1e-12)))
prior_res=res.shift(1)
trigger=(-prior_res).clip(lower=0)
num=pd.DataFrame({a:(trigger[a]*iday[a]).rolling(60,min_periods=45).sum() for a in A})
den=pd.DataFrame({a:trigger[a].rolling(60,min_periods=45).sum() for a in A})
f=num.div(den+1e-12).where(den>1e-7)
print('CANDIDATE lagged_idiosyncratic_downside_to_intraday_recovery_efficiency_60obs endpoint',p.index.max().date(),'assets',len(A),'eligible_dates',int(f.notna().any(axis=1).sum()),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
exec(old[old.index('R={}'):old.index('audit=src[')])
# Regime-stability report using selected best horizon follows library signal audit.
audit=src[src.index('# Full current-library reconstruction'):];exec(audit[:audit.index('mx=-1')])
L['normal_dispersion_residual_downside_upside_magnitude_asymmetry_60obs']=asym
L['normal_dispersion_continuous_downside_residual_rebound_efficiency_60obs']=base
nrk=res.rank(axis=1,pct=True);nd=(nrk.shift(2)-.5).abs();nmv=(nd-(nrk.shift(1)-.5).abs()).where(nd.ge(.30)).where(~high,axis=0);nn=nmv.notna().astype(float).rolling(60,min_periods=20).sum();L['normal_dispersion_residual_extreme_rank_normalization_60obs']=nmv.rolling(60,min_periods=20).mean().where(nn>=20)
aabs=asym.abs();stable=aabs.le(aabs.rolling(60,min_periods=45).median())
L['own_history_stable_asymmetry_normal_dispersion_downside_rebound_efficiency_60obs']=base.where(stable)
L['continuous_asymmetry_penalized_normal_dispersion_downside_rebound_efficiency_60obs']=base*(1-aabs.clip(0,1))
mx=-1;missing=[]
for n,vv in L.items():
 q=pd.concat([f.stack(),vv.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)<8: missing.append(n);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx: mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)
# Print temporal robustness for all horizons under three non-overlapping epochs.
for h,rr in R.items():
 print('REGIMES horizon',h)
 for label,lo,up in [('2026_2029','2026-01-01','2029-12-31'),('2030_2032','2030-01-01','2032-12-31'),('recent_2033_2034','2033-01-01','2034-02-15')]:
  z=rr.loc[lo:up].dropna(); print(label,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/(z.std(ddof=1)+1e-12),6),'hit',round((z>0).mean(),6))
# rank turnover, excluding undefined constant rank days
r=f.rank(axis=1); ts=[]
for t in range(1,len(r)):
 q=pd.concat([r.iloc[t-1],r.iloc[t]],axis=1).dropna()
 if len(q)>=3 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: ts.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('TURNOVER valid_pairs',len(ts),'mean_rank_turnover',round(1-np.mean(ts),6) if ts else None,'median_IQR',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))

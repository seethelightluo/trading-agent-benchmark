"""Miner 2: continuously asymmetry-penalized normal-dispersion rebound efficiency."""
import pandas as pd, numpy as np
old=open('scripts/miner_2_20331013_orthogonal_normal_dispersion_downside_rebound_efficiency_60obs.py').read()
head=old[:old.index("f=pd.DataFrame")].replace("END=pd.Timestamp('2033-10-12')", "END=pd.Timestamp('2033-11-23')")
exec(head)
# Rather than selecting only historically stable-asymmetry states, retain the
# normal-dispersion residual rebound signal continuously and attenuate it by
# each asset's own trailing downside/upside asymmetry magnitude. This produces
# broad, interpretable coverage while testing a smooth state dependence.
aabs=asym.abs()
penalty=1.0-aabs.clip(lower=0,upper=1)
f=base*penalty
print('CANDIDATE continuous_asymmetry_penalized_normal_dispersion_downside_rebound_efficiency_60obs endpoint',p.index.max().date(),'assets',len(A),'eligible_dates',int(f.notna().any(axis=1).sum()),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
exec(old[old.index("R={}"):old.index("audit=src[")])
# Reconstruct the complete currently admitted library using the established
# audit, adding all three rebound-family signals that were admitted after its base.
audit=old[old.index("audit=src["):]
pre=audit[:audit.index('mx=-1')]
exec(pre)
L['normal_dispersion_residual_downside_upside_magnitude_asymmetry_60obs']=asym
L['normal_dispersion_continuous_downside_residual_rebound_efficiency_60obs']=base
nrk=res.rank(axis=1,pct=True);nd=(nrk.shift(2)-.5).abs();nmv=(nd-(nrk.shift(1)-.5).abs()).where(nd.ge(.30)).where(~high,axis=0);nn=nmv.notna().astype(float).rolling(60,min_periods=20).sum();L['normal_dispersion_residual_extreme_rank_normalization_60obs']=nmv.rolling(60,min_periods=20).mean().where(nn>=20)
stable=aabs.le(aabs.rolling(60,min_periods=45).median())
L['own_history_stable_asymmetry_normal_dispersion_downside_rebound_efficiency_60obs']=base.where(stable)
mx=-1;missing=[]
for n,v in L.items():
 q=pd.concat([f.stack(),v.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)<8: missing.append(n); continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if abs(rho)>mx: mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)

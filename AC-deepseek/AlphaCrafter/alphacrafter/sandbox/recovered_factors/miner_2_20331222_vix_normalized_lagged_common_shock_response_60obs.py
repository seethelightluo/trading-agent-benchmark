"""Miner 2: VIX-normalized lagged common-shock residual response, 60 sessions."""
import pandas as pd, numpy as np
old=open('scripts/miner_2_20331208_corrective_market_rebound_efficiency_60obs.py').read()
head=old[:old.index("# A rebound after")].replace("END=pd.Timestamp('2033-12-07')", "END=pd.Timestamp('2033-12-21')")
exec(head)
# Interpretable macro-transmission factor: each asset's idiosyncratic return
# response to yesterday's absolute cross-asset move.  The response is scaled by
# the contemporaneous VIX level relative to its completed 60-day median, so a
# given shock response is distinguished in normalized versus elevated-volatility
# conditions. All terms are available at the close of t.
shock=m.abs().shift(1)
vix=ix('VIX'); vr=vix.pct_change(fill_method=None)
vstate=(vix.shift(1).div(vix.rolling(60,min_periods=45).median().shift(1))-1).clip(-.5,.5)
# 60-session standardized slope, with a minimum of 45 paired observations.
def corr_to(x,y):
 return x.rolling(60,min_periods=45).cov(y).div(x.rolling(60,min_periods=45).std()*y.rolling(60,min_periods=45).std()+1e-12)
raw=pd.DataFrame({a:corr_to(res[a],shock) for a in A})
f=raw.mul(1.0+vstate,axis=0)
print('CANDIDATE vix_normalized_lagged_common_shock_residual_response_60obs endpoint',p.index.max().date(),'assets',len(A),'eligible_dates',int(f.notna().any(axis=1).sum()),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
exec(old[old.index('R={}'):old.index('audit=src[')])
# Exact maintained-library reconstruction used in immediately preceding audit.
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
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'factor',who,'cells',cells,'tested',len(L),'missing',missing)

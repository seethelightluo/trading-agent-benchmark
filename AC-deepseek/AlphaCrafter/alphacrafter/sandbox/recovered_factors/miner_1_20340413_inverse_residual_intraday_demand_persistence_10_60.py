"""Novelty audit for inverse residual intraday demand persistence (10/60)."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
# Use prior maintained-library reconstruction / IC reporting infrastructure.
src=open('scripts/miner_2_20340202_lagged_common_stress_range_expansion_sensitivity_60obs.py').read()
# Its inherited data head contains all base panels and current library reconstruction.
head=src[:src.index("# Price-only liquidity")].replace("END=pd.Timestamp('2034-02-01')", "END=pd.Timestamp('2034-04-12')")
exec(head)
# cross-section median intraday return, 60d rolling market beta, then *invert*
# the 10d residual-demand mean because validation found a negative 20d IC.
intra=p.div(op)-1
common=intra.median(axis=1)
beta=intra.rolling(60,min_periods=45).cov(common).div(common.rolling(60,min_periods=45).var()+1e-12,axis=0)
f=- (intra-beta.mul(common,axis=0)).rolling(10,min_periods=8).mean()
print('CANDIDATE inverse_residual_intraday_demand_persistence_10_60obs endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
# Standard IC/decay measures, ≥8 names per daily cross-sectional observation.
for h in (1,5,10,20):
 fw=p.shift(-h).div(p)-1; z=[]; n=[]
 for d in f.index:
  q=pd.concat([f.loc[d],fw.loc[d]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   v=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(v):z.append((d,v));n.append(len(q))
 x=np.array([v for _,v in z]); print('H',h,'dates',len(x),'mean_names',round(np.mean(n),3),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
 for lab,lo in [('2020_2027','2020-01-01'),('2028_2030','2028-01-01'),('2031_now','2031-01-01'),('latest_6m','2033-10-12')]:
  hi={'2020_2027':'2028-01-01','2028_2030':'2031-01-01','2031_now':'2100-01-01','latest_6m':'2100-01-01'}[lab]
  y=np.array([v for d,v in z if pd.Timestamp(lo)<=d<pd.Timestamp(hi)])
  print(' ',lab,'n',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),6))
r=f.rank(axis=1,pct=True);print('rank_turnover',round(r.diff().abs().stack().mean(),6),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
# Full maintained-library reconstruction inherited from the source, then audit candidate.
audit=src[src.index("audit=src["):]
# Execute reconstruction through before the source's own mx loop.
exec(audit[:audit.index('mx=-1')])
# Later signals added in source after its reconstruction.
L['normal_dispersion_residual_downside_upside_magnitude_asymmetry_60obs']=asym
L['normal_dispersion_continuous_downside_residual_rebound_efficiency_60obs']=base
nrk=res.rank(axis=1,pct=True); nd=(nrk.shift(2)-.5).abs(); nmv=(nd-(nrk.shift(1)-.5).abs()).where(nd.ge(.30)).where(~high,axis=0); nn=nmv.notna().astype(float).rolling(60,min_periods=20).sum(); L['normal_dispersion_residual_extreme_rank_normalization_60obs']=nmv.rolling(60,min_periods=20).mean().where(nn>=20)
aabs=asym.abs(); stable=aabs.le(aabs.rolling(60,min_periods=45).median()); L['own_history_stable_asymmetry_normal_dispersion_downside_rebound_efficiency_60obs']=base.where(stable); L['continuous_asymmetry_penalized_normal_dispersion_downside_rebound_efficiency_60obs']=base*(1-aabs.clip(0,1))
mx=-1; who=None; cells=0; missing=[]
for name,v in L.items():
 q=pd.concat([f.stack(),v.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)<8: missing.append(name);continue
 rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
 if not np.isfinite(rho):missing.append(name);continue
 if abs(rho)>mx: mx=abs(rho);who=name;cells=len(q)
print('AUDIT max_abs_library_correlation',round(mx,6),'closest',who,'evidence_cells',cells,'signals_tested',len(L),'missing',missing)

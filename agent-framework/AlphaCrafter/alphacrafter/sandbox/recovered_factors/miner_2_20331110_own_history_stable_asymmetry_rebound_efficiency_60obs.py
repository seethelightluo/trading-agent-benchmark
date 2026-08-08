"""Miner 2: own-history stable-asymmetry residual rebound efficiency, 60 sessions."""
import pandas as pd, numpy as np
old=open('scripts/miner_2_20331013_orthogonal_normal_dispersion_downside_rebound_efficiency_60obs.py').read()
head=old[:old.index("f=pd.DataFrame")].replace("END=pd.Timestamp('2033-10-12')", "END=pd.Timestamp('2033-11-09')")
exec(head)
# The base signal is retained only when each asset's residual downside/upside
# magnitude asymmetry is no greater than its own trailing median.  Unlike a
# same-day cross-sectional cutoff, this is an interpretable asset-state filter
# and does not mechanically constrain the cross-section to a fixed fraction.
aabs=asym.abs(); stable=aabs.le(aabs.rolling(60,min_periods=45).median())
f=base.where(stable)
print('CANDIDATE own_history_stable_asymmetry_normal_dispersion_downside_rebound_efficiency_60obs endpoint',p.index.max().date(),'assets',len(A),'eligible_dates',int(f.notna().any(axis=1).sum()),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
exec(old[old.index("R={}"):])

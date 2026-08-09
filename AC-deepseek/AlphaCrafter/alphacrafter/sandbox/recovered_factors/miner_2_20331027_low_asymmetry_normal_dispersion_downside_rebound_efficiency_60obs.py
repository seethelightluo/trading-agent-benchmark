"""Miner 2: low-asymmetry-conditioned residual rebound efficiency."""
import pandas as pd, numpy as np
old=open('scripts/miner_2_20331013_orthogonal_normal_dispersion_downside_rebound_efficiency_60obs.py').read()
head=old[:old.index("f=pd.DataFrame")]
head=head.replace("END=pd.Timestamp('2033-10-12')", "END=pd.Timestamp('2033-10-26')")
exec(head)
# A selective, interpretable version of rebound efficiency: retain it only for
# assets whose 60-session downside/upside residual magnitude asymmetry is in the
# lower same-date half. This tests whether rebound effects are distinct when the
# directional asymmetry state is not itself extreme.
ar=asym.rank(axis=1,pct=True)
f=base.where(ar.le(.50))
print('CANDIDATE low_asymmetry_normal_dispersion_downside_rebound_efficiency_60obs endpoint',p.index.max().date(),'assets',len(A),'eligible_dates',int(f.notna().any(axis=1).sum()),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6))
tail=old[old.index("R={}"):]
# ensure reporting title is accurate; inherited reconstruction/audit is otherwise exact.
exec(tail)

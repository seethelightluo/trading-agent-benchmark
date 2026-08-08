"""One idea: inverse extreme broad-upside magnitude-weighted peer-relative capture, full library audit."""
p='scripts/miner_3_20310612_peer_upside_minus_downside_correlation_asymmetry_40.py'
s=open(p).read()
old="""# Candidate: assets whose returns are more correlated with peers in upside than downside sessions have lower downside commonality.
cand=cs(pd.DataFrame({a:r[a].where(m>0).rolling(40,min_periods=12).corr(m.where(m>0))-r[a].where(m<0).rolling(40,min_periods=12).corr(m.where(m<0)) for a in A})).shift(1)"""
new="""# Inverse peer-relative capture on extreme broad-upside dates, weighted by upside magnitude.\n# Signals reversal after unusually strong common upside conditions.\nq75=m.rolling(60,min_periods=40).quantile(.75)\nevent=m>q75\nmag=m.where(event,0)\nnum=rel.mul(mag,axis=0).rolling(60,min_periods=12).sum()\nden=mag.rolling(60,min_periods=12).sum()\nnevent=event.rolling(60,min_periods=40).sum()\ncand=cs((-num.div(den,axis=0)).where(nevent>=12,axis=0)).shift(1)"""
assert old in s
open('scripts/miner_1_20311113_inverse_extreme_broad_upside_capture_60.py','w').write(s.replace(old,new).replace('peer_upside_minus_downside_correlation_asymmetry_40','inverse_extreme_broad_upside_magnitude_weighted_peer_relative_capture_60'))

"""miner_2 one candidate: inverse VIX-conditioned correlation-stress transmission residual."""
from pathlib import Path
src=Path('scripts/miner_2_20320513_inverse_systemic_correlation_stress_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2032-05-12')", "E=pd.Timestamp('2032-05-26')")
old="""# Systemic-correlation shock: mean 20-session pairwise return correlation, standardized
# against its own trailing 60 sessions.  The factor is inverse 30-observation beta
# change to this shock in elevated versus normal-correlation states, residualized
# from own volatility, peer-correlation, downside-beta asymmetry, and trend.
pairs=[R[a].rolling(20,min_periods=12).corr(R[b]) for i,a in enumerate(A) for b in A[i+1:]]
corrstress=pd.concat(pairs,axis=1).mean(axis=1)
cz=(corrstress-corrstress.rolling(60,min_periods=40).mean())/(corrstress.rolling(60,min_periods=40).std()+1e-12)
high=cz>=cz.rolling(60,min_periods=40).median()
F=res(-(beta(cz,high)-beta(cz,~high)),v,peer,dba,trend)"""
new="""# Candidate: mean 20-session pairwise correlation standardized versus 60 sessions.
# Separate each asset's correlation-shock beta between independently observable VIX
# shock sessions (VIX return > its rolling 60-session median) and other sessions.
# Inverting the difference favors assets with less correlation-stress transmission
# specifically when volatility information confirms a market fear shock.
pairs=[R[a].rolling(20,min_periods=12).corr(R[b]) for i,a in enumerate(A) for b in A[i+1:]]
corrstress=pd.concat(pairs,axis=1).mean(axis=1)
cz=(corrstress-corrstress.rolling(60,min_periods=40).mean())/(corrstress.rolling(60,min_periods=40).std()+1e-12)
vixret=rd('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
vixhigh=vixret>=vixret.rolling(60,min_periods=40).median()
F=res(-(beta(cz,vixhigh)-beta(cz,~vixhigh)),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new).replace('inverse_systemic_correlation_stress_transmission_residual_30','inverse_vix_conditioned_correlation_stress_transmission_residual_30')
# repair text label occurring in print replacement is fine
exec(compile(src,'miner_2_vix_corrstress_20320527','exec'))
"""
# VIX is observation-only and is used solely as a conditioning signal. All factor inputs
# are clipped to the supplied prior-session endpoint; no symbols outside A are traded.
"""
Path('scripts/miner_2_20320527_inverse_vix_conditioned_correlation_stress_transmission_residual_30.py').write_text(src)
print('wrote candidate script')

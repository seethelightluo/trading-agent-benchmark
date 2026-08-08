"""miner_2 validation: inverse systemic correlation-stress transmission residual."""
from pathlib import Path
src=Path('scripts/miner_3_20320318_inverse_global_dispersion_shock_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2032-03-17')", "E=pd.Timestamp('2032-05-12')")
old="""disp=R.std(axis=1)
dz=(disp-disp.rolling(60,min_periods=40).mean())/(disp.rolling(60,min_periods=40).std()+1e-12)
high=dz>=dz.rolling(60,min_periods=40).median()
F=res(-(beta(dz,high)-beta(dz,~high)),v,peer,dba,trend)"""
new="""# Systemic-correlation shock: mean 20-session pairwise return correlation, standardized
# against its own trailing 60 sessions.  The factor is inverse 30-observation beta
# change to this shock in elevated versus normal-correlation states, residualized
# from own volatility, peer-correlation, downside-beta asymmetry, and trend.
pairs=[R[a].rolling(20,min_periods=12).corr(R[b]) for i,a in enumerate(A) for b in A[i+1:]]
corrstress=pd.concat(pairs,axis=1).mean(axis=1)
cz=(corrstress-corrstress.rolling(60,min_periods=40).mean())/(corrstress.rolling(60,min_periods=40).std()+1e-12)
high=cz>=cz.rolling(60,min_periods=40).median()
F=res(-(beta(cz,high)-beta(cz,~high)),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new).replace('inverse_global_dispersion_shock_transmission_residual_30','inverse_systemic_correlation_stress_transmission_residual_30')
exec(compile(src,'miner_2_systemic_corrstress_20320513','exec'))

"""Miner_2: inverse commodity-rate transmission asymmetry residual, one candidate."""
from pathlib import Path
src=Path('scripts/miner_3_20320318_inverse_global_dispersion_shock_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2032-03-17')", "E=pd.Timestamp('2032-08-18')")
old="""# Candidate: cross-asset dispersion shock is the daily standard deviation of all
# tradable-asset returns, standardized versus its trailing 60 sessions.  The score is
# the *inverse* 30-observation beta difference during high-dispersion versus ordinary
# sessions, residualized from generic volatility, peer crowding, downside-beta
# asymmetry and trend. Higher values indicate unusually low transmission of a broad
# disagreement/shock regime, rather than simply low standalone volatility.
disp=R.std(axis=1)
dz=(disp-disp.rolling(60,min_periods=40).mean())/(disp.rolling(60,min_periods=40).std()+1e-12)
high=dz>=dz.rolling(60,min_periods=40).median()
F=res(-(beta(dz,high)-beta(dz,~high)),v,peer,dba,trend)"""
new="""# Candidate: a copper-minus-oil daily return is a real-growth versus energy-cost
# shock.  Estimate each asset's 30-session beta to this shock separately when the
# 10-year US yield is rising and falling.  The inverse beta difference is then
# residualized from generic volatility, peer crowding, downside-market beta asymmetry,
# and trend. Higher scores represent relative resilience to growth/energy shocks when
# the discount-rate regime changes, not a generic commodity beta or trend signal.
co=R['COPPER']-R['WTI']
yup=R['US10Y']>=0
F=res(-(beta(co,yup)-beta(co,~yup)),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new)
src=src.replace('inverse_global_dispersion_shock_transmission_residual_30','inverse_copper_oil_shock_rate_regime_transmission_residual_30')
# Do not allow self-comparison to the replaced generic-dispersion candidate: it is not active.
src=src.replace("'dispersion_conditioned_reversal'=", "'dispersion_conditioned_reversal'=")
exec(compile(src,'miner_2_copper_oil_rate_20320819','exec'))

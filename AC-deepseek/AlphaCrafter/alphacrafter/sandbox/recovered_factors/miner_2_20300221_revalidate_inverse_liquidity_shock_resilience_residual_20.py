"""Miner_2 scheduled revalidation: inverse liquidity-shock resilience residual, as of 2030-02-20.
Single-factor validation with reconstructed full active-library novelty screen."""
from pathlib import Path
src=Path('scripts/miner_2_20300207_continuous_participation_weighted_rebound_residual_20.py').read_text()
src=src.replace("E=pd.Timestamp('2030-02-06')", "E=pd.Timestamp('2030-02-20')")
src=src.replace("continuous_participation_weighted_rebound_residual_20", "inverse_liquidity_shock_resilience_residual_20")
old="""# Candidate: participation-adaptive downside recovery. Mean current return following a prior
# loss, on above-normal own-volume sessions where volume exists; structural volume absence
# uses an unconditional fallback. Residualization removes risk, crowding, downside beta and trend.
V=pd.DataFrame({a:rd(a,'volume') for a in A})
relvol=V/V.rolling(20,min_periods=15).mean()
wt=relvol.clip(0.5,2.0).fillna(1.0)
priorloss=(-R.shift(1)/(v.shift(1)+1e-12)).clip(0,4)
raw=R.mul(priorloss*wt).rolling(20,min_periods=10).sum().div((priorloss*wt).rolling(20,min_periods=10).sum().replace(0,np.nan),axis=0)/(v+1e-12)
F=res(raw,v,peer,dba,P/P.shift(20)-1)"""
new="""# Revalidated candidate: inverse liquidity-shock resilience. On sessions with own
# relative volume above its trailing 60-observation 75th percentile, average return is
# calculated over 20 observations and scaled by realized volatility. Residualization removes
# 20-day trend, risk and peer crowding; the inverse orientation is the admitted direction.
V=pd.DataFrame({a:rd(a,'volume') for a in A})
relvol=V/(V.rolling(20,min_periods=15).mean()+1e-12)
liquid_shock=relvol>=relvol.rolling(60,min_periods=40).quantile(.75)
raw=R.where(liquid_shock).rolling(20,min_periods=6).mean()/(v+1e-12)
F=-res(raw,P/P.shift(20)-1,v,peer)
# Reconstruct the newly admitted participation-weighted recovery signal for novelty.
_wt=relvol.clip(.5,2).fillna(1); _pl=(-R.shift(1)/(v.shift(1)+1e-12)).clip(0,4)
_part=R.mul(_pl*_wt).rolling(20,min_periods=10).sum().div((_pl*_wt).rolling(20,min_periods=10).sum().replace(0,np.nan),axis=0)/(v+1e-12)
participation_rebound=res(_part,v,peer,dba,P/P.shift(20)-1)"""
assert old in src
src=src.replace(old,new)
# Add the factor missing from the 7-Feb library snapshot to ensure all 24 other active signals are tested.
needle="L['inverse_usdjpy_shock_transmission_beta_asymmetry_residual_30']=res(-(_u-_d),v,peer,dba,P/P.shift(20)-1)"
src=src.replace(needle,needle+"\nL['continuous_participation_weighted_rebound_residual_20']=participation_rebound")
Path('scripts/miner_2_20300221_revalidate_inverse_liquidity_shock_resilience_residual_20.py').write_text(src)
print('written')

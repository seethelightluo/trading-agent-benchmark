"""Miner_2 one-idea validation: liquidity-shock resilience residual, visible through 2029-12-12."""
from pathlib import Path
src=Path('scripts/miner_2_20291115_volume_confirmed_recovery_quality_residual_20.py').read_text()
src=src.replace("E=pd.Timestamp('2029-11-14')", "E=pd.Timestamp('2029-12-12')")
src=src.replace('volume_confirmed_recovery_quality_residual_20','liquidity_shock_resilience_residual_20')
old="""# Candidate: volume-confirmed recovery quality. A recent five-day rebound is more credible when
# its own trading volume is above its trailing 20-day baseline.  Residualizing the
# volatility-scaled interaction against 20-day trend, risk and peer crowding isolates
# confirmation rather than generic momentum or liquidity level.
V=pd.DataFrame({a:rd(a,'volume') for a in A})
relvol=V/(V.rolling(20,min_periods=15).mean()+1e-12)
raw=(P/P.shift(5)-1)/(v+1e-12)*np.log1p(relvol.clip(lower=0))
F=res(raw,P/P.shift(20)-1,v,peer)"""
new="""# Candidate: liquidity-shock resilience.  An asset's return on sessions when its own
# volume is unusually elevated (above its trailing 60-observation 75th percentile) is averaged
# over the latest 20 observations and volatility-scaled.  Positive values identify assets that
# absorb high-participation liquidity shocks constructively. Residualization against 20-day
# trend, realized risk and peer crowding separates this from generic momentum, low volatility,
# and liquidity level.
V=pd.DataFrame({a:rd(a,'volume') for a in A})
relvol=V/(V.rolling(20,min_periods=15).mean()+1e-12)
liquid_shock=relvol>=relvol.rolling(60,min_periods=40).quantile(.75)
raw=R.where(liquid_shock).rolling(20,min_periods=6).mean()/(v+1e-12)
F=res(raw,P/P.shift(20)-1,v,peer)"""
assert old in src
src=src.replace(old,new)
exec(compile(src,'miner_2_liquidity_shock_resilience','exec'))

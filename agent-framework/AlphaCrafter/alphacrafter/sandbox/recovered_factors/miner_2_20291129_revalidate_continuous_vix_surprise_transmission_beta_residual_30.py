"""Miner_2 periodic revalidation: continuous VIX-surprise transmission beta residual, visible through 2029-11-28."""
from pathlib import Path
src=Path('scripts/miner_2_20291115_volume_confirmed_recovery_quality_residual_20.py').read_text()
src=src.replace("E=pd.Timestamp('2029-11-14')", "E=pd.Timestamp('2029-11-28')")
src=src.replace('volume_confirmed_recovery_quality_residual_20','continuous_vix_surprise_transmission_beta_residual_30')
old="""# Candidate: volume-confirmed recovery quality. A recent five-day rebound is more credible when
# its own trading volume is above its trailing 20-day baseline.  Residualizing the
# volatility-scaled interaction against 20-day trend, risk and peer crowding isolates
# confirmation rather than generic momentum or liquidity level.
V=pd.DataFrame({a:rd(a,'volume') for a in A})
relvol=V/(V.rolling(20,min_periods=15).mean()+1e-12)
raw=(P/P.shift(5)-1)/(v+1e-12)*np.log1p(relvol.clip(lower=0))
F=res(raw,P/P.shift(20)-1,v,peer)"""
new="""# Revalidation candidate: continuous VIX-surprise transmission asymmetry. Standardize
# VIX returns by trailing 60-observation volatility and estimate 30-observation asset betas
# separately on above- versus below-median VIX surprises. Negative beta asymmetry scores
# resilience to escalating volatility, residualized from generic volatility, crowding,
# downside-market beta asymmetry, and 20-observation trend.
vixr=rd('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
z=vixr/(vixr.rolling(60,min_periods=40).std()+1e-12)
pos=z.where(z>=z.rolling(60,min_periods=40).median())
neg=z.where(z<z.rolling(60,min_periods=40).median())
up=pd.DataFrame({a:R[a].where(pos.notna()).rolling(30,min_periods=10).cov(pos)/pos.rolling(30,min_periods=10).var() for a in A})
dn=pd.DataFrame({a:R[a].where(neg.notna()).rolling(30,min_periods=10).cov(neg)/neg.rolling(30,min_periods=10).var() for a in A})
F=res(-(up-dn),v,peer,dba,P/P.shift(20)-1)"""
assert old in src
src=src.replace(old,new)
exec(compile(src,'miner_2_revalidate_continuous_vix','exec'))

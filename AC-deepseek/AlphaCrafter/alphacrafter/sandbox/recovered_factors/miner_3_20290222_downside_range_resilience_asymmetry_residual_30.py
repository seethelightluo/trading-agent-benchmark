"""Miner_3 one-idea research: downside-range resilience asymmetry, 30 observations.
Higher score identifies assets with relatively compressed intraday ranges on broad market-down days,
net of conventional trend, volatility, peer-correlation, and downside-beta-asymmetry effects.
All inputs are restricted to the visible cutoff."""
from pathlib import Path
src=Path('scripts/miner_3_20281116_yield_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2028-11-15')", "E=pd.Timestamp('2029-02-21')")
src=src.replace('yield_shock_transmission_beta_asymmetry_residual_30', 'downside_range_resilience_asymmetry_residual_30')
old="""# Candidate: asymmetry of 30-observation exposure to US 10-year yield changes. A high score
# identifies assets that respond differently to rising versus falling yields, residualized from
# broad-market beta asymmetry, volatility, crowding, and medium-term trend.
yld=R['US10Y']
rise=pd.DataFrame({a:R[a].where(yld>0).rolling(30,min_periods=10).cov(yld.where(yld>0))/yld.where(yld>0).rolling(30,min_periods=10).var() for a in A})
fall=pd.DataFrame({a:R[a].where(yld<0).rolling(30,min_periods=10).cov(yld.where(yld<0))/yld.where(yld<0).rolling(30,min_periods=10).var() for a in A})
F=res(rise-fall,v,peer,dba,P/P.shift(20)-1)"""
new="""# Candidate: conditional intraday range resilience. Normalizing high-low range by close
# makes the measure comparable across heterogeneous price scales. Higher values mean an asset's
# range is lower on aggregate-market down days than on market-up days, a potential stress-resilience
# characteristic rather than a generic low-volatility signal after residualization.
H=pd.DataFrame({a:rd(a,'high') for a in A}); Lo=pd.DataFrame({a:rd(a,'low') for a in A})
rng=(H-Lo).div(P.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
downrng=rng.where(M<0,axis=0).rolling(30,min_periods=10).mean()
uprng=rng.where(M>0,axis=0).rolling(30,min_periods=10).mean()
F=res(uprng-downrng,v,peer,dba,P/P.shift(20)-1)"""
if old not in src: raise RuntimeError('replacement anchor not found')
src=src.replace(old,new)
exec(compile(src,'downside_range_resilience_candidate','exec'))

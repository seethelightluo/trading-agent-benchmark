"""Miner_3 one-idea test: market-down intraday range stability residual, 30 obs, cutoff 2029-01-10.
Higher score means an asset's normalized daily range is relatively smaller on broad market-down
sessions than on market-up sessions, after removing ordinary volatility/crowding/beta/trend and
close-location resilience. Tests whether stress-session price containment predicts subsequent returns.
"""
from pathlib import Path
src=Path('scripts/miner_3_20281116_yield_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2028-11-15')", "E=pd.Timestamp('2029-01-10')")
src=src.replace('yield_shock_transmission_beta_asymmetry_residual_30', 'market_down_intraday_range_stability_residual_30')
old="""# Candidate: asymmetry of 30-observation exposure to US 10-year yield changes. A high score
# identifies assets that respond differently to rising versus falling yields, residualized from
# broad-market beta asymmetry, volatility, crowding, and medium-term trend.
yld=R['US10Y']
rise=pd.DataFrame({a:R[a].where(yld>0).rolling(30,min_periods=10).cov(yld.where(yld>0))/yld.where(yld>0).rolling(30,min_periods=10).var() for a in A})
fall=pd.DataFrame({a:R[a].where(yld<0).rolling(30,min_periods=10).cov(yld.where(yld<0))/yld.where(yld<0).rolling(30,min_periods=10).var() for a in A})
F=res(rise-fall,v,peer,dba,P/P.shift(20)-1)"""
new="""# Candidate: market-direction conditional intraday range asymmetry.  A high score is
# low normalized range on broad market-down sessions relative to market-up sessions, i.e.
# price containment specifically in cross-asset stress.  It is residualized from the ordinary
# volatility level, peer crowding, downside beta asymmetry, trend and down-day close location.
O=pd.DataFrame({a:rd(a,'open') for a in A});H=pd.DataFrame({a:rd(a,'high') for a in A});Lo=pd.DataFrame({a:rd(a,'low') for a in A})
rng=(H-Lo).div(P.shift(1)).replace([np.inf,-np.inf],np.nan)
downrng=rng.where(M<0).rolling(30,min_periods=10).mean()
uprng=rng.where(M>0).rolling(30,min_periods=10).mean()
loc=(P-Lo)/(H-Lo).replace(0,np.nan)
downloc=loc.where(M<0).rolling(20,min_periods=6).mean()
F=res(uprng-downrng,v,peer,dba,P/P.shift(20)-1,downloc)"""
if old not in src: raise RuntimeError('candidate replacement failed')
src=src.replace(old,new)
exec(compile(src,'market_down_intraday_range_stability_generated','exec'))
"""

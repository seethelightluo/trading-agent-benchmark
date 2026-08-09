"""Miner_3: downside peer-correlation resilience residual (30 observations), cutoff 2028-12-27.
One idea: assets whose average correlation to peers is lower during equal-weight-market down days
than during up days may offer diversification precisely when cross-asset stress occurs."""
from pathlib import Path
src=Path('scripts/miner_3_20281116_yield_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2028-11-15')", "E=pd.Timestamp('2028-12-27')")
src=src.replace('yield_shock_transmission_beta_asymmetry_residual_30', 'downside_peer_correlation_resilience_residual_30')
old="""# Candidate: asymmetry of 30-observation exposure to US 10-year yield changes. A high score
# identifies assets that respond differently to rising versus falling yields, residualized from
# broad-market beta asymmetry, volatility, crowding, and medium-term trend.
yld=R['US10Y']
rise=pd.DataFrame({a:R[a].where(yld>0).rolling(30,min_periods=10).cov(yld.where(yld>0))/yld.where(yld>0).rolling(30,min_periods=10).var() for a in A})
fall=pd.DataFrame({a:R[a].where(yld<0).rolling(30,min_periods=10).cov(yld.where(yld<0))/yld.where(yld<0).rolling(30,min_periods=10).var() for a in A})
F=res(rise-fall,v,peer,dba,P/P.shift(20)-1)"""
new="""# Candidate: peer-correlation asymmetry conditional on cross-asset market direction.
# Higher raw values mean correlation with the other 14 assets is lower on market-down than
# market-up days; residualization prevents this from merely replicating unconditional crowding,
# volatility, market downside beta asymmetry, or medium-term trend.
downcorr=pd.DataFrame({a:pd.concat([R[a].where(M<0).rolling(30,min_periods=10).corr(R[b].where(M<0)) for b in A if b!=a],axis=1).mean(1) for a in A})
upcorr=pd.DataFrame({a:pd.concat([R[a].where(M>0).rolling(30,min_periods=10).corr(R[b].where(M>0)) for b in A if b!=a],axis=1).mean(1) for a in A})
F=res(upcorr-downcorr,v,peer,dba,P/P.shift(20)-1)"""
if old not in src: raise RuntimeError('candidate replacement failed')
src=src.replace(old,new)
exec(compile(src,'downside_peer_correlation_candidate_generated','exec'))

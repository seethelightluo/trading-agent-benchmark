"""Miner_3: VIX-shock transmission beta asymmetry residual, 30 observations, visible 2028-11-29.
High score: an asset's beta to rising VIX changes less its beta to falling VIX changes,
orthogonalized to unconditional VIX beta, volatility, peer crowding, market downside-beta asymmetry and trend.
"""
# This candidate is intentionally a one-idea variant of the established beta-asymmetry framework.
from pathlib import Path
src=Path('scripts/miner_3_20281116_yield_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2028-11-15')", "E=pd.Timestamp('2028-11-29')")
src=src.replace('yield_shock_transmission_beta_asymmetry_residual_30', 'vix_shock_transmission_beta_asymmetry_residual_30')
old="""# Candidate: asymmetry of 30-observation exposure to US 10-year yield changes. A high score
# identifies assets that respond differently to rising versus falling yields, residualized from
# broad-market beta asymmetry, volatility, crowding, and medium-term trend.
yld=R['US10Y']
rise=pd.DataFrame({a:R[a].where(yld>0).rolling(30,min_periods=10).cov(yld.where(yld>0))/yld.where(yld>0).rolling(30,min_periods=10).var() for a in A})
fall=pd.DataFrame({a:R[a].where(yld<0).rolling(30,min_periods=10).cov(yld.where(yld<0))/yld.where(yld<0).rolling(30,min_periods=10).var() for a in A})
F=res(rise-fall,v,peer,dba,P/P.shift(20)-1)"""
new="""# Candidate: asymmetry of 30-observation exposure to VIX percentage changes.  The score
# measures differing participation in volatility shocks versus volatility relief, after removing
# unconditional VIX beta and established broad cross-asset risk characteristics.
vix_chg=rd('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
rise=pd.DataFrame({a:R[a].where(vix_chg>0).rolling(30,min_periods=10).cov(vix_chg.where(vix_chg>0))/vix_chg.where(vix_chg>0).rolling(30,min_periods=10).var() for a in A})
fall=pd.DataFrame({a:R[a].where(vix_chg<0).rolling(30,min_periods=10).cov(vix_chg.where(vix_chg<0))/vix_chg.where(vix_chg<0).rolling(30,min_periods=10).var() for a in A})
uncond=pd.DataFrame({a:R[a].rolling(30,min_periods=15).corr(vix_chg) for a in A})
F=res(rise-fall,uncond,v,peer,dba,P/P.shift(20)-1)"""
if old not in src: raise RuntimeError('candidate replacement failed')
src=src.replace(old,new)
exec(compile(src, 'vix_shock_candidate_generated', 'exec'))

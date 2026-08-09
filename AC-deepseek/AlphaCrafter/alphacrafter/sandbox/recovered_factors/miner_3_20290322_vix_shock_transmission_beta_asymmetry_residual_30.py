"""One-idea validation: 30d VIX-shock transmission beta asymmetry residual."""
from pathlib import Path
src=Path('scripts/miner_3_20281116_yield_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2028-11-15')", "E=pd.Timestamp('2029-03-21')")
src=src.replace('yield_shock_transmission_beta_asymmetry_residual_30','vix_shock_transmission_beta_asymmetry_residual_30')
old="""# Candidate: asymmetry of 30-observation exposure to US 10-year yield changes. A high score
# identifies assets that respond differently to rising versus falling yields, residualized from
# broad-market beta asymmetry, volatility, crowding, and medium-term trend.
yld=R['US10Y']
rise=pd.DataFrame({a:R[a].where(yld>0).rolling(30,min_periods=10).cov(yld.where(yld>0))/yld.where(yld>0).rolling(30,min_periods=10).var() for a in A})
fall=pd.DataFrame({a:R[a].where(yld<0).rolling(30,min_periods=10).cov(yld.where(yld<0))/yld.where(yld<0).rolling(30,min_periods=10).var() for a in A})
F=res(rise-fall,v,peer,dba,P/P.shift(20)-1)"""
new="""# Candidate: asymmetric transmission of VIX shocks.  For each asset estimate its 30-day
# return beta separately on VIX-increase and VIX-decrease days.  The score is the negative
# (risk-off beta minus risk-on beta), so high scores identify assets comparatively resilient
# during volatility shocks; remove generic vol, crowding, market-asymmetry and trend effects.
vixret=rd('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
up=pd.DataFrame({a:R[a].where(vixret>0).rolling(30,min_periods=10).cov(vixret.where(vixret>0))/vixret.where(vixret>0).rolling(30,min_periods=10).var() for a in A})
dn=pd.DataFrame({a:R[a].where(vixret<0).rolling(30,min_periods=10).cov(vixret.where(vixret<0))/vixret.where(vixret<0).rolling(30,min_periods=10).var() for a in A})
F=res(-(up-dn),v,peer,dba,P/P.shift(20)-1)"""
if old not in src: raise RuntimeError('anchor not found')
src=src.replace(old,new)
exec(compile(src,'vix_shock_transmission_candidate','exec'))

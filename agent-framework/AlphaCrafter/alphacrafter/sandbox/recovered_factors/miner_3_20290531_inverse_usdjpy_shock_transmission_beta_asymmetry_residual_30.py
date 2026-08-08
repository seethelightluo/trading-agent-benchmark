"""One idea validation: inverse USDJPY shock-transmission beta-asymmetry residual, cutoff 2029-05-30."""
from pathlib import Path
src=Path('scripts/miner_3_20290322_vix_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2029-03-21')", "E=pd.Timestamp('2029-05-30')")
src=src.replace('vix_shock_transmission_beta_asymmetry_residual_30','inverse_usdjpy_shock_transmission_beta_asymmetry_residual_30')
old="""# Candidate: asymmetric transmission of VIX shocks.  For each asset estimate its 30-day
# return beta separately on VIX-increase and VIX-decrease days.  The score is the negative
# (risk-off beta minus risk-on beta), so high scores identify assets comparatively resilient
# during volatility shocks; remove generic vol, crowding, market-asymmetry and trend effects.
vixret=rd('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
up=pd.DataFrame({a:R[a].where(vixret>0).rolling(30,min_periods=10).cov(vixret.where(vixret>0))/vixret.where(vixret>0).rolling(30,min_periods=10).var() for a in A})
dn=pd.DataFrame({a:R[a].where(vixret<0).rolling(30,min_periods=10).cov(vixret.where(vixret<0))/vixret.where(vixret<0).rolling(30,min_periods=10).var() for a in A})
F=res(-(up-dn),v,peer,dba,P/P.shift(20)-1)"""
new="""# Candidate: opposite orientation of USDJPY conditional beta asymmetry. A high score
# identifies stronger beta to yen-weakening than yen-strengthening shocks, after residualizing
# generic volatility, peer crowding, market downside asymmetry and trend.
fx=rd('USDJPY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
up=pd.DataFrame({a:R[a].where(fx>0).rolling(30,min_periods=10).cov(fx.where(fx>0))/fx.where(fx>0).rolling(30,min_periods=10).var() for a in A})
dn=pd.DataFrame({a:R[a].where(fx<0).rolling(30,min_periods=10).cov(fx.where(fx<0))/fx.where(fx<0).rolling(30,min_periods=10).var() for a in A})
F=res(up-dn,v,peer,dba,P/P.shift(20)-1)"""
if old not in src: raise RuntimeError('candidate anchor missing')
src=src.replace(old,new)
exec(compile(src,'inverse_usdjpy_shock_candidate','exec'))

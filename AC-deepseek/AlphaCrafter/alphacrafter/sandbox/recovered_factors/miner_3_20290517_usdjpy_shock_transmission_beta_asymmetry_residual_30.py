"""One idea: 30d USDJPY-shock transmission beta asymmetry residual, cutoff 2029-05-16."""
from pathlib import Path
src=Path('scripts/miner_3_20290419_oil_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2029-04-18')", "E=pd.Timestamp('2029-05-16')")
src=src.replace('oil_shock_transmission_beta_asymmetry_residual_30','usdjpy_shock_transmission_beta_asymmetry_residual_30')
old="""# Candidate: asymmetric transmission of WTI oil shocks. For each asset estimate its
# 30-day return beta separately on WTI-up and WTI-down days. The negative beta difference
# ranks assets that are relatively resilient to adverse oil declines versus oil rallies.
# Cross-sectionally residualize generic volatility, peer crowding, market downside-beta
# asymmetry, and 20-day trend, leaving an interpretable energy-shock-specific component.
oilret=R['WTI']
up=pd.DataFrame({a:R[a].where(oilret>0).rolling(30,min_periods=10).cov(oilret.where(oilret>0))/oilret.where(oilret>0).rolling(30,min_periods=10).var() for a in A})
dn=pd.DataFrame({a:R[a].where(oilret<0).rolling(30,min_periods=10).cov(oilret.where(oilret<0))/oilret.where(oilret<0).rolling(30,min_periods=10).var() for a in A})
F=res(-(up-dn),v,peer,dba,P/P.shift(20)-1)"""
new="""# Candidate: asymmetric transmission of USDJPY shocks. Estimate each asset's 30-day
# return beta separately on yen-weakening and yen-strengthening days; the negative difference
# favors assets that retain comparatively favorable exposure when the yen strengthens. Generic
# volatility, peer crowding, market downside-asymmetry and 20-day trend are residualized.
fx=rd('USDJPY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
up=pd.DataFrame({a:R[a].where(fx>0).rolling(30,min_periods=10).cov(fx.where(fx>0))/fx.where(fx>0).rolling(30,min_periods=10).var() for a in A})
dn=pd.DataFrame({a:R[a].where(fx<0).rolling(30,min_periods=10).cov(fx.where(fx<0))/fx.where(fx<0).rolling(30,min_periods=10).var() for a in A})
F=res(-(up-dn),v,peer,dba,P/P.shift(20)-1)"""
if old not in src: raise RuntimeError('candidate anchor missing')
src=src.replace(old,new)
# Add the active post-baseline signals to the mandatory correlation screen.
needle="""L['downside_market_yield_hedge_beta_residual_30']=res(_bd-_bn,v,peer,dba,P/P.shift(20)-1)"""
addition=needle+"""
# admitted: downside-market crypto transmission residual
_c=R['BTC']; _bdc=pd.DataFrame({a:R[a].where(_md).rolling(30,min_periods=10).cov(_c.where(_md))/_c.where(_md).rolling(30,min_periods=10).var() for a in A}); _bnc=pd.DataFrame({a:R[a].where(~_md).rolling(30,min_periods=10).cov(_c.where(~_md))/_c.where(~_md).rolling(30,min_periods=10).var() for a in A}); L['downside_market_crypto_transmission_beta_residual_30']=res(_bdc-_bnc,v,peer,dba,P/P.shift(20)-1)
# admitted: continuous VIX surprise transmission residual
_vs=rd('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index); _vp=pd.DataFrame({a:R[a].rolling(30,min_periods=10).cov(_vs)/_vs.rolling(30,min_periods=10).var() for a in A}); L['continuous_vix_surprise_transmission_beta_residual_30']=res(_vp,v,peer,dba,P/P.shift(20)-1)
# admitted oil-shock beta-asymmetry residual
_o=R['WTI']; _ou=pd.DataFrame({a:R[a].where(_o>0).rolling(30,min_periods=10).cov(_o.where(_o>0))/_o.where(_o>0).rolling(30,min_periods=10).var() for a in A}); _od=pd.DataFrame({a:R[a].where(_o<0).rolling(30,min_periods=10).cov(_o.where(_o<0))/_o.where(_o<0).rolling(30,min_periods=10).var() for a in A}); L['oil_shock_transmission_beta_asymmetry_residual_30']=res(-(_ou-_od),v,peer,dba,P/P.shift(20)-1)"""
if needle not in src: raise RuntimeError('library anchor missing')
src=src.replace(needle,addition)
exec(compile(src,'usdjpy_shock_candidate','exec'))

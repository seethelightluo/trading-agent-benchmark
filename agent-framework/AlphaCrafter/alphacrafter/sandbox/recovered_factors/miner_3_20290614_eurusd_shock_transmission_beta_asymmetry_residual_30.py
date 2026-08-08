"""One idea: EURUSD conditional shock-transmission beta asymmetry residual; cutoff 2029-06-13."""
from pathlib import Path
src=Path('scripts/miner_3_20290531_inverse_usdjpy_shock_transmission_beta_asymmetry_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2029-05-30')", "E=pd.Timestamp('2029-06-13')")
src=src.replace('inverse_usdjpy_shock_transmission_beta_asymmetry_residual_30','eurusd_shock_transmission_beta_asymmetry_residual_30')
old="""# Candidate: opposite orientation of USDJPY conditional beta asymmetry. A high score
# identifies stronger beta to yen-weakening than yen-strengthening shocks, after residualizing
# generic volatility, peer crowding, market downside asymmetry and trend.
fx=rd('USDJPY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
up=pd.DataFrame({a:R[a].where(fx>0).rolling(30,min_periods=10).cov(fx.where(fx>0))/fx.where(fx>0).rolling(30,min_periods=10).var() for a in A})
dn=pd.DataFrame({a:R[a].where(fx<0).rolling(30,min_periods=10).cov(fx.where(fx<0))/fx.where(fx.where(fx<0).rolling(30,min_periods=10).var() for a in A})
F=res(up-dn,v,peer,dba,P/P.shift(20)-1)"""
# exact source uses valid denominator; avoid fragile replacement anchors
start=src.index('# Candidate: opposite orientation of USDJPY')
end=src.index('\n# Reconstructions',start)
new="""# Candidate: asymmetric transmission of EURUSD shocks. Higher scores indicate assets
# with stronger beta when EURUSD rises than when it falls, independent of generic risk,
# peer crowding, market downside asymmetry, and medium-term trend. EURUSD is observation-only.
fx=rd('EURUSD',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
up=pd.DataFrame({a:R[a].where(fx>0).rolling(30,min_periods=10).cov(fx.where(fx>0))/fx.where(fx>0).rolling(30,min_periods=10).var() for a in A})
dn=pd.DataFrame({a:R[a].where(fx<0).rolling(30,min_periods=10).cov(fx.where(fx<0))/fx.where(fx<0).rolling(30,min_periods=10).var() for a in A})
F=res(up-dn,v,peer,dba,P/P.shift(20)-1)"""
src=src[:start]+new+src[end:]
# Extend older reconstruction screen with all factors admitted since its base template.
needle="print('FACTOR eurusd_shock_transmission_beta_asymmetry_residual_30"
insert="""# Completeness additions: current macro-transmission factors reconstructed under identical residual controls.
def asym(sig, sign=1):
 u=pd.DataFrame({a:R[a].where(sig>0).rolling(30,min_periods=10).cov(sig.where(sig>0))/sig.where(sig>0).rolling(30,min_periods=10).var() for a in A})
 d=pd.DataFrame({a:R[a].where(sig<0).rolling(30,min_periods=10).cov(sig.where(sig<0))/sig.where(sig<0).rolling(30,min_periods=10).var() for a in A})
 return res(sign*(u-d),v,peer,dba,P/P.shift(20)-1)
L['continuous_vix_surprise_transmission_beta_residual_30']=asym(rd('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index),-1)
L['downside_market_crypto_transmission_beta_residual_30']=asym(R['BTC'])
L['oil_shock_transmission_beta_asymmetry_residual_30']=asym(R['WTI'])
L['inverse_usdjpy_shock_transmission_beta_asymmetry_residual_30']=asym(rd('USDJPY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index))
"""
pos=src.index(needle)
src=src[:pos]+insert+src[pos:]
exec(compile(src,'eurusd_shock_candidate','exec'))
"""

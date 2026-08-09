"""Miner_2 one-idea validation: gold-copper divergence transmission asymmetry residual."""
from pathlib import Path
src=Path('scripts/miner_2_20290419_vix_shock_conditional_transmission_beta_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2029-04-18')", "E=pd.Timestamp('2029-10-03')")
src=src.replace('vix_shock_conditional_transmission_beta_residual_30','gold_copper_divergence_transmission_beta_residual_30')
old="""# Candidate: VIX-shock conditional transmission.  On sessions whose VIX return is in
# its trailing-60-day upper quartile, estimate each asset's 30-observation beta to VIX return,
# less its beta on all other sessions.  The negative differential ranks resilience to abrupt
# volatility repricing.  Residualization removes generic volatility, crowding, downside-market
# beta asymmetry and trend; this makes the signal a conditional transmission factor, not a
# static low-volatility or VIX-beta factor.
vixr=rd('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
shockstate=vixr>=vixr.rolling(60,min_periods=40).quantile(.75)
hi=pd.DataFrame({a:R[a].where(shockstate).rolling(30,min_periods=10).cov(vixr.where(shockstate))/vixr.where(shockstate).rolling(30,min_periods=10).var() for a in A})
lo=pd.DataFrame({a:R[a].where(~shockstate).rolling(30,min_periods=10).cov(vixr.where(~shockstate))/vixr.where(~shockstate).rolling(30,min_periods=10).var() for a in A})
F=res(-(hi-lo),v,peer,dba,P/P.shift(20)-1)"""
new="""# Candidate: gold-copper divergence transmission asymmetry.  The driver is the daily
# return spread XAU minus COPPER, a liquid cross-asset macro proxy for defensive-versus-growth
# repricing.  For each asset, compare 30-day betas on positive and negative spread days; the
# negative difference ranks resilience when gold outperforms copper.  Residualize static risk,
# crowding, market-down beta asymmetry, and trend to isolate this state-dependent component.
drv=R['XAU']-R['COPPER']
up=pd.DataFrame({a:R[a].where(drv>0).rolling(30,min_periods=10).cov(drv.where(drv>0))/drv.where(drv>0).rolling(30,min_periods=10).var() for a in A})
dn=pd.DataFrame({a:R[a].where(drv<0).rolling(30,min_periods=10).cov(drv.where(drv<0))/drv.where(drv<0).rolling(30,min_periods=10).var() for a in A})
F=res(-(up-dn),v,peer,dba,P/P.shift(20)-1)"""
assert old in src
src=src.replace(old,new)
# Append the three factors admitted after the inherited library snapshot, ensuring all active signals are screened.
needle="print('FACTOR gold_copper_divergence_transmission_beta_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L));ics={}"
addition="""# Later admitted active factors, reconstructed for the mandatory full-library screen.
_vr=rd('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index); _z=_vr/(_vr.rolling(60,min_periods=40).std()+1e-12); _p=_z.where(_z>=_z.rolling(60,min_periods=40).median()); _n=_z.where(_z<_z.rolling(60,min_periods=40).median())
_u=pd.DataFrame({a:R[a].where(_p.notna()).rolling(30,min_periods=10).cov(_p)/_p.rolling(30,min_periods=10).var() for a in A}); _d=pd.DataFrame({a:R[a].where(_n.notna()).rolling(30,min_periods=10).cov(_n)/_n.rolling(30,min_periods=10).var() for a in A}); L['continuous_vix_surprise_transmission_beta_residual_30']=res(-(_u-_d),v,peer,dba,P/P.shift(20)-1)
_o=R['WTI']; _u=pd.DataFrame({a:R[a].where(_o>0).rolling(30,min_periods=10).cov(_o.where(_o>0))/_o.where(_o>0).rolling(30,min_periods=10).var() for a in A}); _d=pd.DataFrame({a:R[a].where(_o<0).rolling(30,min_periods=10).cov(_o.where(_o<0))/_o.where(_o<0).rolling(30,min_periods=10).var() for a in A}); L['oil_shock_transmission_beta_asymmetry_residual_30']=res(-(_u-_d),v,peer,dba,P/P.shift(20)-1)
_j=rd('USDJPY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index); _u=pd.DataFrame({a:R[a].where(_j>0).rolling(30,min_periods=10).cov(_j.where(_j>0))/_j.where(_j>0).rolling(30,min_periods=10).var() for a in A}); _d=pd.DataFrame({a:R[a].where(_j<0).rolling(30,min_periods=10).cov(_j.where(_j<0))/_j.where(_j<0).rolling(30,min_periods=10).var() for a in A}); L['inverse_usdjpy_shock_transmission_beta_asymmetry_residual_30']=res(-(_u-_d),v,peer,dba,P/P.shift(20)-1)
print('FACTOR gold_copper_divergence_transmission_beta_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L));ics={}"""
assert needle in src
src=src.replace(needle,addition)
exec(compile(src,'miner2_gold_copper_divergence','exec'))

"""Miner_2 one-idea validation: inverse FX-disagreement shock transmission residual."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-11-12')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Pre-specified candidate: inverse transmission to cross-currency disagreement.
# The observation-only USDJPY and USDCNY daily changes are each standardized by
# their trailing 60-session volatility; their signed difference measures a
# relative Asian-FX repricing rather than a common USD move. The signal is the
# negative 30-session beta to this continuous, lag-free observable shock,
# residualized daily against volatility, peer crowding, downside beta asymmetry
# and 20-session trend. Higher scores mean lower relative-FX shock sensitivity.
jpy=rd('USDJPY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
cny=rd('USDCNY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
shock=jpy/(jpy.rolling(60,min_periods=40).std()+1e-12)-cny/(cny.rolling(60,min_periods=40).std()+1e-12)
F=res(-beta(shock,shock.notna()),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','inverse_fx_disagreement_shock_transmission_residual_30')
needle="print('FACTOR inverse_fx_disagreement_shock_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))"
# Add all post-base admitted families to the inherited broad reconstruction.
extra="""# Point-in-time reconstruction of admitted post-base signal families for novelty.
cb=R[['WTI','COPPER','XAU']].mean(1); eb=R[['SPX','NDX','SX5E','HSI']].mean(1); div=cb-eb
L['inverse_commodity_equity_divergence']=res(-(beta(div,div>=0)-beta(div,div<0)),v,peer,dba,trend)
rs=R['US10Y']-R['CN10Y']; rsz=rs/(rs.rolling(60,min_periods=40).std()+1e-12)
L['inverse_continuous_rate_spread_surprise']=res(-beta(rsz,rsz.notna()),v,peer,dba,trend)
st=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,3)/3
L['inverse_equity_stress_rate_transmission']=-res(beta(R['US10Y']*st,pd.Series(True,index=P.index))-beta(R['US10Y']*(1-st),pd.Series(True,index=P.index)),v,peer,dba,trend)
print('FACTOR inverse_fx_disagreement_shock_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))"""
assert needle in src
src=src.replace(needle,extra)
exec(compile(src,'miner_2_inverse_fx_disagreement_20311113','exec'))

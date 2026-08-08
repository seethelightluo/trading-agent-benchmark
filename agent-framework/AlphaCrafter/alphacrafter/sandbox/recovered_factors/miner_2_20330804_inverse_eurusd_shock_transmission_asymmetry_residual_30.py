"""Miner_2 single-candidate validation: inverse EURUSD shock-transmission asymmetry residual (30 sessions)."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2033-08-03')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: inverse difference between exposure to positive and negative EURUSD
# daily shocks. It measures assets that respond relatively defensively when the dollar
# weakens versus strengthens, after removing volatility, correlation crowding, market
# downside-beta asymmetry, and 20-session trend. EURUSD is observation-only.
fx=rd('EURUSD',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
F=-res(beta(fx,fx>0)-beta(fx,fx<0),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new)
src=src.replace("print('FACTOR oil_market_drawdown_conditional_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))", "print('FACTOR inverse_eurusd_shock_transmission_asymmetry_residual_30 visible_through',E.date(),'assets',len(A),'library_proxy_signals',len(L))")
exec(compile(src,'miner_2_eurusd_shock_asymmetry_20330804','exec'))

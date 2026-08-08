"""miner_2 single-idea validation: rate-volatility-state transmission asymmetry."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-03-05')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: 30-day sensitivity to US10Y changes in elevated versus quiet
# rate-volatility states. The state is lagged, based only on the rolling 20-day
# absolute US10Y return relative to its prior 60-day median. Higher scores mean
# an asset becomes relatively more rate-sensitive when the rates market is noisy.
# Residualization removes unconditional trend, own volatility, peer crowding and
# broad-market downside-beta asymmetry.
fx=R['US10Y']
rate_state=fx.abs().rolling(20,min_periods=15).mean().shift(1) > fx.abs().rolling(60,min_periods=40).mean().shift(1)
F=res(beta(fx,rate_state)-beta(fx,~rate_state),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new)
src=src.replace("print('FACTOR oil_market_drawdown_conditional_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))", "print('FACTOR rate_volatility_state_transmission_asymmetry_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))")
exec(compile(src,'rate_volatility_state_transmission_asymmetry_residual_30','exec'))

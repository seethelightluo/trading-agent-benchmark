"""Miner_2: single-candidate test, inverse rate-currency co-shock transmission residual."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2033-03-30')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: inverse beta to a continuous rate-currency co-shock: the product of
# standardized USDCNY and US10Y daily surprises.  Positive factor values favor assets
# relatively resilient when CNY and US-rate shocks occur jointly, after removing
# generic volatility, peer crowding, market beta asymmetry and trend.
cny=rd('USDCNY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
yld=R['US10Y']
cnyz=cny/(cny.rolling(60,min_periods=40).std()+1e-12)
yldz=yld/(yld.rolling(60,min_periods=40).std()+1e-12)
fx=cnyz*yldz
F=-res(beta(fx,fx.notna(),30,15),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new)
src=src.replace("print('FACTOR oil_market_drawdown_conditional_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))", "print('FACTOR inverse_rate_currency_coshock_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_proxy_signals',len(L))")
exec(compile(src,'rate_currency_coshock_20330331','exec'))

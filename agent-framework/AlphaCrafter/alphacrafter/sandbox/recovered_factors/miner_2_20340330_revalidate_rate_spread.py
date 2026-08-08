"""Fixed-specification point-in-time revalidation through 2034-03-15."""
from pathlib import Path
src=Path('scripts/miner_3_20320415_revalidate_inverse_rate_spread_30.py').read_text()
src=src.replace("E=pd.Timestamp('2032-04-14')", "E=pd.Timestamp('2034-03-15')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Fixed admitted construction, no parameter or orientation changes.
spread=R['US10Y']-R['CN10Y']
fx=spread/(spread.rolling(60,min_periods=40).std()+1e-12)
F=res(-beta(fx,fx.notna()),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new).replace("FACTOR oil_market_drawdown_conditional_transmission_residual_30", "FACTOR inverse_continuous_rate_spread_surprise_transmission_residual_30_REVALIDATION")
exec(compile(src,'rate_spread_revalidation_20340330','exec'))

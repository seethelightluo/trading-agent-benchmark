"""Miner_2 one-idea exploration: lagged range-expansion recovery residual."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-09-03')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: lagged range-expansion recovery. Measure each asset's intraday
# close location, then average it on sessions preceded by an own true-range
# expansion (range divided by trailing 20-session range, minus one). A high
# score means the asset repeatedly closes strongly after its unusually wide
# range days. Residualize cross-sectionally from volatility, peer crowding,
# downside beta asymmetry, and 20-session trend.
O=pd.DataFrame({a:rd(a,'open') for a in A});H=pd.DataFrame({a:rd(a,'high') for a in A});Lo=pd.DataFrame({a:rd(a,'low') for a in A})
rng=(H-Lo)/P.replace(0,np.nan); ex=(rng/rng.rolling(20,min_periods=15).mean()-1).shift(1).clip(lower=0,upper=4)
loc=(P-Lo)/(H-Lo).replace(0,np.nan)
raw=loc.mul(ex).rolling(20,min_periods=10).sum().div(ex.rolling(20,min_periods=10).sum().replace(0,np.nan))
F=res(raw,v,peer,dba,trend)"""
if old not in src: raise RuntimeError('candidate anchor absent')
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','lagged_range_expansion_recovery_residual_20')
exec(compile(src,'miner_2_range_expansion_recovery_20310904','exec'))

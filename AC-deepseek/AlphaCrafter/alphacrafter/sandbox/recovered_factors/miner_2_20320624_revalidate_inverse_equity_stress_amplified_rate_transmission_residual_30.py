"""Fixed-specification point-in-time revalidation through 2032-06-23."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2032-06-23')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Fixed admitted factor: inverse residualized difference between 30-session beta to
# US10Y changes amplified by lagged equity stress and beta to rate changes in calm.
stress=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,3)/3
rate=R['US10Y']
F=-res(beta(rate*stress,pd.Series(True,index=P.index))-beta(rate*(1-stress),pd.Series(True,index=P.index)),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','equity_stress_amplified_rate_transmission_residual_30')
exec(compile(src,'miner_2_equity_stress_rate_revalidation_20320624','exec'))

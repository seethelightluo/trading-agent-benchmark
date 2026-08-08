"""Miner_2 candidate validation: continuous own-drawdown recovery residual, point-in-time.
One interpretable candidate: returns following deeper lagged own drawdowns, weighted
continuously by lagged broad-equity stress, then cross-sectionally residualized."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2033-02-02')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: continuous own-drawdown recovery.  An asset receives a high score if
# its subsequent daily returns have been stronger following its own deeper lagged
# drawdowns, with observations emphasized only as lagged broad equity stress rises.
# Controls remove generic volatility, crowding, downside beta asymmetry and trend.
equity=R[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1)
stress=(-equity.shift(1)/(equity.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,3)/3
dd=P/P.rolling(20,min_periods=15).max()-1
weight=(-dd.shift(1)/(v.shift(1)+1e-12)).clip(0,5).mul(stress,axis=0)
raw=R.mul(weight).rolling(20,min_periods=12).sum().div(weight.rolling(20,min_periods=12).sum().replace(0,np.nan),axis=0)
F=res(raw/(v+1e-12),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','continuous_own_drawdown_stress_recovery_residual_20')
# Ensure title identifies candidate
src=src.replace("FACTOR oil_market_drawdown_conditional_transmission_residual_30", "FACTOR continuous_own_drawdown_stress_recovery_residual_20")
Path('scripts/miner_2_20330203_continuous_own_drawdown_stress_recovery_residual_20.py').write_text(src)
exec(compile(src,'miner_2_20330203_candidate','exec'))

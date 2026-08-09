"""Miner_2 one-idea exploration: relative-volume-conditioned recovery residual.
Uses only data visible through 2031-09-03.  The candidate is each asset's
20-session volatility-normalized return following an above-normal own-volume
day, residualized from common cross-asset risk characteristics.
"""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-09-03')")
src=src.replace("P=pd.DataFrame({a:rd(a) for a in A});R=P.pct_change(fill_method=None);M=R.mean(1);v=R.rolling(20,min_periods=15).std();v5=R.rolling(5,min_periods=4).std()", "P=pd.DataFrame({a:rd(a) for a in A}); V=pd.DataFrame({a:rd(a,'volume') for a in A}); R=P.pct_change(fill_method=None);M=R.mean(1);v=R.rolling(20,min_periods=15).std();v5=R.rolling(5,min_periods=4).std()")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: relative-volume-conditioned recovery. At t, a lagged own-volume
# surprise is volume / trailing-20-day volume minus one, clipped to [-2, 4].
# The signal is the 20-session weighted mean of next-day-own returns following
# those surprises, scaled by own volatility.  It asks whether an asset's
# unusually active sessions have been followed by persistent relative recovery,
# beyond volatility, crowding, downside-beta asymmetry, and trend.
vs=(V/V.rolling(20,min_periods=15).mean()-1).shift(1).clip(-2,4)
raw=R.mul(vs.clip(lower=0)).rolling(20,min_periods=10).sum().div(vs.clip(lower=0).rolling(20,min_periods=10).sum().replace(0,np.nan))/(v+1e-12)
F=res(raw,v,peer,dba,trend)"""
if old not in src: raise RuntimeError('candidate anchor absent')
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','relative_volume_conditioned_recovery_residual_20')
exec(compile(src,'miner_2_relative_volume_recovery_20310904','exec'))

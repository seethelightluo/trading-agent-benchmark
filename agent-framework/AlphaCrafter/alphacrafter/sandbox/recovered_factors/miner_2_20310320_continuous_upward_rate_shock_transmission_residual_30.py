"""miner_2 single-idea validation: continuous rate-shock transmission beta residual."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-03-19')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: continuous rate-shock transmission. Standardize completed US10Y
# changes by its lagged 60-session volatility, and estimate each asset's 30-session
# beta to the positive tail minus beta to nonpositive rate moves. The continuous
# shock definition preserves observations relative to sparse volatility state splits.
# A positive score indicates stronger exposure specifically to unusually upward
# rate moves, net of own risk, peer crowding, downside-market beta and trend.
rawrate=R['US10Y']
shock=rawrate/(rawrate.rolling(60,min_periods=40).std().shift(1)+1e-12)
pos=shock.ge(0)
F=res(beta(shock,pos)-beta(shock,~pos),v,peer,dba,trend)"""
if old not in src: raise RuntimeError('candidate anchor absent')
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','continuous_upward_rate_shock_transmission_residual_30')
exec(compile(src,'miner_2_continuous_upward_rate_shock_transmission_20310320','exec'))

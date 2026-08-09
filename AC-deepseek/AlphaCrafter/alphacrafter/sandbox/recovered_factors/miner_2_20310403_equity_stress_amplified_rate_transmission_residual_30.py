"""miner_2 single-idea validation: equity-stress-amplified rate transmission residual."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-04-02')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: equity-stress-amplified rate transmission. A lagged, continuous
# broad-market stress intensity is the prior day's negative equal-weight market
# return scaled by trailing 60-day market volatility and clipped to [0,3]. For
# each asset, take its 30-session beta to US10Y changes amplified by stress,
# less beta to the complementary calm-state rate changes. Higher values denote
# relatively stronger rate transmission when equity stress is elevated, net of
# own volatility, peer crowding, downside beta asymmetry and trend.
stress=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,3)/3
rate=R['US10Y']
F=res(beta(rate*stress,pd.Series(True,index=P.index))-beta(rate*(1-stress),pd.Series(True,index=P.index)),v,peer,dba,trend)"""
if old not in src: raise RuntimeError('candidate anchor absent')
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','equity_stress_amplified_rate_transmission_residual_30')
exec(compile(src,'miner_2_equity_stress_rate_transmission_20310403','exec'))

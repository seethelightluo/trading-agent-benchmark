"""Miner_2 single-candidate validation: inverse dispersion-amplified US-China rate-spread transmission residual."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2033-04-27')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: inverse sensitivity to a US-China rate-spread surprise, with exposure
# amplified only when lagged cross-asset dispersion is unusually high. This distinguishes
# heterogeneous macro repricing from ordinary rate moves, then removes generic risk,
# crowding, downside beta asymmetry and trend.
spread=R['US10Y']-R['CN10Y']
spz=spread/(spread.rolling(60,min_periods=40).std()+1e-12)
disp=R.std(axis=1)
dz=((disp-disp.rolling(60,min_periods=40).mean())/(disp.rolling(60,min_periods=40).std()+1e-12)).clip(0,3)
fx=spz*(1+dz.shift(1))
F=-res(beta(fx,fx.notna(),30,15),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new)
src=src.replace("print('FACTOR oil_market_drawdown_conditional_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))", "print('FACTOR inverse_dispersion_amplified_us_cn_rate_spread_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_proxy_signals',len(L))")
exec(compile(src,'inverse_dispersion_rate_spread_20330428','exec'))

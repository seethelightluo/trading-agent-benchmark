"""Miner_2 single-candidate validation: inverse defensive-metal relative-shock transmission residual."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2033-09-14')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: inverse beta to the defensive-metal relative shock (gold return less copper return),
# observed on broad-market down days. Higher values identify assets insulated from a flight-to-
# safety versus growth-material repricing during equity stress. Residualization removes generic
# volatility, peer crowding, broad downside-beta asymmetry and trend.
fx=R['XAU']-R['COPPER']
F=-res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new)
src=src.replace("print('FACTOR oil_market_drawdown_conditional_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))", "print('FACTOR inverse_defensive_metal_relative_shock_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_proxy_signals',len(L))")
exec(compile(src,'inverse_defensive_metal_relative_shock_20330915','exec'))

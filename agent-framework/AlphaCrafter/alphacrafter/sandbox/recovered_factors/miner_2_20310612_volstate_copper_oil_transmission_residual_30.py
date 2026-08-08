"""miner_2: validate volatility-state conditioned copper-oil transmission residual."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-06-11')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: relative growth-versus-energy (copper-minus-oil) shock beta in
# volatility expansion versus compression states. Higher scores identify assets
# whose commodity-relative transmission rises as cross-asset volatility expands,
# after removal of generic volatility, crowding, downside-beta asymmetry and trend.
fx=R['COPPER']-R['WTI']
state=M.rolling(5,min_periods=4).std() >= M.rolling(60,min_periods=40).std()
F=res(beta(fx,state)-beta(fx,~state),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new)
src=src.replace("FACTOR oil_market_drawdown_conditional_transmission_residual_30", "FACTOR volstate_copper_oil_transmission_residual_30")
exec(compile(src,'miner_2_volstate_copper_oil_20310612','exec'))

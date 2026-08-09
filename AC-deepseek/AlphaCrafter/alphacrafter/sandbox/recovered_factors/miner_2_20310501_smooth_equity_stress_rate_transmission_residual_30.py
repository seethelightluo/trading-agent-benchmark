"""miner_2 single-idea validation: smooth equity-stress rate-transmission residual."""
from pathlib import Path
src=Path('scripts/miner_2_20310403_equity_stress_amplified_rate_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2031-04-02')", "E=pd.Timestamp('2031-04-30')")
src=src.replace('equity_stress_amplified_rate_transmission_residual_30','smooth_equity_stress_rate_transmission_residual_30')
old="""stress=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,3)/3
rate=R['US10Y']
F=res(beta(rate*stress,pd.Series(True,index=P.index))-beta(rate*(1-stress),pd.Series(True,index=P.index)),v,peer,dba,trend)"""
new="""# Smooth stress weight: lagged negative broad-market return in units of its
# trailing 60-session volatility, transformed x/(1+x). Unlike the predecessor's
# clipped linear tier, every negative-stress observation contributes gradually;
# the complement is the calm rate component. Signal is inverse-oriented after
# preliminary raw-rate transmission evidence was negative.
x=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(lower=0)
stress=x/(1+x)
rate=R['US10Y']
F=-res(beta(rate*stress,pd.Series(True,index=P.index))-beta(rate*(1-stress),pd.Series(True,index=P.index)),v,peer,dba,trend)"""
if old not in src: raise RuntimeError('candidate anchor absent')
src=src.replace(old,new)
exec(compile(src,'miner_2_smooth_equity_stress_rate_transmission_20310501','exec'))

"""Miner_2 one-idea validation: inverse rate-spread shock transmission residual (30)."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-10-01')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: inverse transmission to an unexpected widening in the US-versus-CN
# 10-year yield-return spread. The shock is the lagged standardized absolute
# rate-spread return, and signal is minus the 30-session beta conditional on
# broad-market weakness. It identifies assets relatively resilient to large
# cross-country rate repricings in risk-off sessions, after removing generic
# volatility, crowding, downside beta asymmetry, and trend.
spread=R['US10Y']-R['CN10Y']
shock=(spread.abs()/ (spread.abs().rolling(60,min_periods=40).std()+1e-12)).shift(1).clip(0,5)
F=-res(beta(shock,M<0,30,10),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','inverse_rate_spread_shock_transmission_residual_30')
# add miner_2's known admitted factor to the library screen
needle="print('FACTOR inverse_rate_spread_shock_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))"
add="""stress=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,3)/3
L['inverse_equity_stress_amplified_rate_transmission_residual_30']=-res(beta(R['US10Y']*stress,pd.Series(True,index=P.index))-beta(R['US10Y']*(1-stress),pd.Series(True,index=P.index)),v,peer,dba,trend)
"""
assert needle in src
src=src.replace(needle,add+needle)
# broad chronological regimes, at same 5d evaluation horizon
oldreg="for n,m in []:\n x=ics[1][m];print('regime',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')"
newreg="for n,m in [('2020_23',ics[5].index<'2024-01-01'),('2024_27',(ics[5].index>='2024-01-01')&(ics[5].index<'2028-01-01')),('2028_current',ics[5].index>='2028-01-01')]:\n x=ics[5][m];print('regime5',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')"
assert oldreg in src
src=src.replace(oldreg,newreg)
exec(compile(src,'miner_2_inverse_rate_spread_shock_20311002','exec'))

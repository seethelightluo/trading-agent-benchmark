"""Miner_2 validation: continuous US-CN rate-spread surprise transmission residual.
Includes the previously admitted equity-stress rate-transmission signal in the
reconstructed library comparison, unlike the base harness.
"""
from pathlib import Path
src=Path('scripts/miner_3_20310710_continuous_rate_spread_surprise_transmission_residual_30.py').read_text()
needle="print('FACTOR continuous_rate_spread_surprise_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L));ics={}"
replacement="""# Add the admitted miner_2 state-conditioned rate signal to complete the library screen.
stress=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,3)/3
old_factor=-res(beta(R['US10Y']*stress,pd.Series(True,index=P.index))-beta(R['US10Y']*(1-stress),pd.Series(True,index=P.index)),v,peer,dba,trend)
L['inverse_equity_stress_amplified_rate_transmission_residual_30']=old_factor
print('FACTOR continuous_rate_spread_surprise_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L));ics={}"""
assert needle in src
src=src.replace(needle,replacement)
exec(compile(src,'miner_2_continuous_rate_spread_surprise_20310724','exec'))

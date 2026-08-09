"""Miner 1 research: continuous DXY beta migration, endpoint 2035-01-03.
The signal is each asset's 20-session beta to daily DXY changes minus its
60-session beta. It measures a changing currency transmission channel without
using direction buckets or contemporaneous future data.
"""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-01-03')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""dxy_signal=macro('DXY')
f=beta(r,dxy_signal,20,mp=15)-beta(r,dxy_signal,60,mp=45)
print('CANDIDATE continuous_dxy_beta_migration_20_60obs')"""
assert old in s
s=s.replace(old,new)
# Avoid overwriting the later dxy assignment; keep it equivalent to the candidate input.
s=s.replace("vix=macro('VIX');dxy=macro('DXY');negday=r<0", "vix=macro('VIX');dxy=dxy_signal;negday=r<0")
exec(compile(s,'miner_1_continuous_dxy_beta_migration','exec'))

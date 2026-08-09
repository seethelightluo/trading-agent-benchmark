"""Candidate: residual directional range asymmetry acceleration, 20/60 observations.
Continuous relative change in idiosyncratic downside versus upside intraday range."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-02-28')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""rng=(hi-lo).div(p).replace(0,np.nan)
# A rise means downside-session range has widened relative to upside-session range
# versus its own slower baseline; negative values indicate relative downside compression.
down=rng.where(res<0).rolling(20,min_periods=8).mean()
up=rng.where(res>=0).rolling(20,min_periods=8).mean()
down60=rng.where(res<0).rolling(60,min_periods=22).mean()
up60=rng.where(res>=0).rolling(60,min_periods=22).mean()
f=np.log(down/up)-np.log(down60/up60)
print('CANDIDATE residual_directional_range_asymmetry_acceleration_20_60obs')"""
assert old in s
s=s.replace(old,new)
exec(compile(s,'miner_1_residual_directional_range_asymmetry_acceleration','exec'))

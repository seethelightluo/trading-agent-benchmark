"""Candidate: trend-orthogonal residual-weighted continuous close-location pressure acceleration."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-03-28')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Continuous intraday pressure: close location is weighted by absolute idiosyncratic move,
# then measured as a 20-versus-60-session acceleration and cross-sectionally stripped of trend.
pressure=(clv-.5)*res.abs().div(v.replace(0,np.nan)).clip(upper=4)
raw=pressure.rolling(20,min_periods=15).mean()-pressure.rolling(60,min_periods=45).mean()
trend=(p/p.shift(20)-1)/v
f=orth(raw,trend)
print('CANDIDATE trend_orthogonal_residual_weighted_continuous_close_location_pressure_acceleration_20_60obs')"""
assert old in s
s=s.replace(old,new)
# A zero-variance comparator is not correlation evidence: print each audit as supplied by contract.
exec(compile(s,'miner_1_residual_weighted_continuous_clv_pressure','exec'))
"""
"""

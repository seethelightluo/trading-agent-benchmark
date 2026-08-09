"""Miner 1 candidate: trend-orthogonal continuous gap-day reversal acceleration."""
src=open('scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py').read()
src=src.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2034-07-05')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Continuous reversal strength: positive when a gap and the same-day move oppose,
# weighted by their magnitudes; its short-versus-long acceleration is trend-orthogonal.
gap=op/p.shift(1)-1
day=p/op-1
raw=(-(gap*day)).rolling(10,min_periods=8).mean()-(-(gap*day)).rolling(60,min_periods=45).mean()
trend=(p/p.shift(20)-1)/v
f=orth(raw,trend)
print('FACTOR trend_orthogonal_gap_day_reversal_acceleration_10_60_20: acceleration in continuous overnight-gap versus daytime-return reversal strength, cross-sectionally residualized to 20d risk-adjusted trend')"""
assert old in src
src=src.replace(old,new)
src=src.replace('continuous close-location pressure.', 'trend-orthogonal continuous gap-day reversal acceleration.')
src=src.replace("print('AUDIT endpoint'", "print('CANDIDATE trend_orthogonal_gap_day_reversal_acceleration_10_60_20 endpoint'")
exec(src)

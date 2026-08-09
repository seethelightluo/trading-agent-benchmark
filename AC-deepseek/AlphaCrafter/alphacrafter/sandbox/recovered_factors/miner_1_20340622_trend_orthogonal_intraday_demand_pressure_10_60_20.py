"""Miner 1: trend-orthogonal intraday demand pressure (10 vs 60 observations)."""
src=open('scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py').read()
src=src.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2034-06-21')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""raw=(p/op-1).rolling(10,min_periods=8).mean()-(p/op-1).rolling(60,min_periods=45).mean()
# Remove each date's linear cross-sectional exposure to 20-day risk-adjusted trend.
trend=(p/p.shift(20)-1)/v
f=orth(raw,trend)
print('FACTOR trend_orthogonal_intraday_demand_pressure_10_60_20: short-minus-long close/open demand, cross-sectionally residualized to 20d risk-adjusted trend')"""
assert old in src
src=src.replace(old,new)
src=src.replace('continuous close-location pressure.', 'trend-orthogonal intraday demand pressure.')
src=src.replace("print('AUDIT endpoint'", "print('CANDIDATE trend_orthogonal_intraday_demand_pressure_10_60_20 endpoint'")
exec(src)

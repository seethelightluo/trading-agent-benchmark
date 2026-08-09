"""Validate one candidate: residual-downside range-weighted close-location recovery (10/60)."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-08-29')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# On idiosyncratic down sessions, range-weight close location. A positive short-vs-long
# change denotes progressively better intraday absorption of unusually large residual losses.
down_range=((hi-lo).div(p).where(res<0)).clip(lower=0)
num10=clv.mul(down_range).rolling(10,min_periods=4).sum(); den10=down_range.rolling(10,min_periods=4).sum()
num60=clv.mul(down_range).rolling(60,min_periods=18).sum(); den60=down_range.rolling(60,min_periods=18).sum()
f=num10/den10-num60/den60
print('CANDIDATE residual_downside_range_weighted_close_location_recovery_10_60obs')"""
assert old in s
s=s.replace(old,new)
s=s.replace("if H==10:","if H==5:")
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)","('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'residual_downside_range_weighted_close_location_recovery','exec'))

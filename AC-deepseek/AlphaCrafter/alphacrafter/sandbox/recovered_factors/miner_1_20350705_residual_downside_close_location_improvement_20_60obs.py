"""Single-candidate validation: residual downside close-location improvement, 20 vs 60 observations."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-07-04')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# On idiosyncratically down sessions, improvement in closing location over its
# medium-term level is a continuous measure of selling absorption.
down_clv=clv.where(res<0)
f=down_clv.rolling(20,min_periods=8).mean()-down_clv.rolling(60,min_periods=20).mean()
print('CANDIDATE residual_downside_close_location_improvement_20_60obs')"""
assert old in s
s=s.replace(old,new)
s=s.replace("if H==10:","if H==20:")
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)","('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'residual_downside_close_location_improvement','exec'))

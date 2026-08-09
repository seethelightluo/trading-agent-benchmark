"""Candidate: residual-downside range-compression / close-location concordance, 10/60 observations.
Tests whether compressed idiosyncratic downside ranges jointly with improving closes on
those sessions signal a cross-asset repair, independently audited vs active library."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-02-14')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""downrange=(hi-lo).div(p).where(res<0)
range_compression=-np.log(downrange.rolling(10,min_periods=4).mean()/downrange.rolling(60,min_periods=18).mean())
down_clv=clv.where(res<0)
clv_improve=down_clv.rolling(10,min_periods=4).mean()-down_clv.rolling(60,min_periods=18).mean()
f=range_compression*clv_improve
print('CANDIDATE residual_downside_range_compression_close_location_concordance_10_60obs')"""
assert old in s
s=s.replace(old,new)
exec(compile(s,'miner_1_residual_downside_range_compression_close_location_concordance','exec'))

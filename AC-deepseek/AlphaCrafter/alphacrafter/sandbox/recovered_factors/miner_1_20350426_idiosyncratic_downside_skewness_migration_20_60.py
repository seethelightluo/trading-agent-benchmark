"""Candidate validation: idiosyncratic downside-skewness migration, full-library audit."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-04-25')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Idiosyncratic downside-skewness migration.  This compares recent residual
# downside-tail asymmetry to its own medium-run state, standardized by total
# residual variability. A rising value indicates an asset whose idiosyncratic
# left tail has recently become unusually pronounced; negative orientation
# expresses the interpretable tail-exhaustion/reversal hypothesis.
neg=res.clip(upper=0)
recent=neg.pow(3).rolling(20,min_periods=15).mean().div(res.pow(2).rolling(20,min_periods=15).mean().pow(1.5))
base_skew=neg.pow(3).rolling(60,min_periods=45).mean().div(res.pow(2).rolling(60,min_periods=45).mean().pow(1.5))
f=-(recent-base_skew)
print('CANDIDATE idiosyncratic_downside_skewness_migration_20_60')"""
assert old in s
s=s.replace(old,new)
exec(compile(s,'miner_1_idiosyncratic_downside_skewness_migration_20_60','exec'))

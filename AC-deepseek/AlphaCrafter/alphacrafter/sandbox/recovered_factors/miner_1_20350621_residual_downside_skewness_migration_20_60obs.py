"""Single-candidate validation: residual downside-skewness migration (20 vs 60 days)."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-06-20')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Negative change in residual downside skewness: unusually worsening left-tail
# asymmetry relative to its medium-term baseline is treated as exhaustion.
neg=res.clip(upper=0)
recent=neg.pow(3).rolling(20,min_periods=15).mean().div(res.pow(2).rolling(20,min_periods=15).mean().pow(1.5))
base=neg.pow(3).rolling(60,min_periods=45).mean().div(res.pow(2).rolling(60,min_periods=45).mean().pow(1.5))
f=-(recent-base)
print('CANDIDATE residual_downside_skewness_migration_20_60obs')"""
assert old in s
s=s.replace(old,new)
# make the legacy printed regime segmentation more informative through current endpoint
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)","('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'residual_downside_skewness_migration','exec'))

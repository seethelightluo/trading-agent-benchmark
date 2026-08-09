"""Candidate validation: residual US10Y beta migration (20-versus-60 observations)."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-06-06')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Change in residual exposure to daily US10Y moves, comparing recent to
# medium-run trailing windows. Positive values identify a relative migration
# toward rate sensitivity, orthogonal to market-direction and price trend.
y10=r['US10Y']
f=beta(res,y10,20,mp=15)-beta(res,y10,60,mp=45)
print('CANDIDATE residual_us10y_beta_migration_20_60obs')"""
assert old in s
s=s.replace(old,new)
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)", "('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'miner_1_residual_us10y_beta_migration_20_60obs','exec'))

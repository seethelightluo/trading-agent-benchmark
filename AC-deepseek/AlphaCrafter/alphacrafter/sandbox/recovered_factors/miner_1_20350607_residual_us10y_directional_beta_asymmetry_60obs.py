"""Candidate: residual US10Y rate-direction beta asymmetry (60 observations)."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-06-06')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Cross-asset residual sensitivity difference between rising and falling US10Y
# sessions. This measures a persistent, direction-specific rates exposure after
# removal of the daily common return, rather than price trend or OHLC location.
# US10Y is a tradable benchmark series but used here solely as the rate-state leg.
y10=r['US10Y']
f=beta(res,y10.where(y10>0),60,mp=16)-beta(res,y10.where(y10<=0),60,mp=16)
print('CANDIDATE residual_us10y_directional_beta_asymmetry_60obs')"""
assert old in s
s=s.replace(old,new)
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)", "('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'miner_1_residual_us10y_directional_beta_asymmetry_60obs','exec'))

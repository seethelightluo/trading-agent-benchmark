"""Candidate validation: VIX beta migration (20-versus-60 sessions), full library audit."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-05-09')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# VIX beta migration measures a change in an asset's idiosyncratic sensitivity
# to daily VIX innovations. Positive values identify assets becoming relatively
# more defensive during recent risk-price moves than in their medium-run state.
# The residual return removes common cross-asset market motion before beta fitting.
vix_m=macro('VIX')
f=beta(res,vix_m,20,mp=15)-beta(res,vix_m,60,mp=45)
print('CANDIDATE residual_vix_beta_migration_20_60obs')"""
assert old in s
s=s.replace(old,new)
exec(compile(s,'miner_1_residual_vix_beta_migration_20_60obs','exec'))

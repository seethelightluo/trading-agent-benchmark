"""Single idea: continuous US10Y beta migration (20 versus 60 sessions).
Each asset's short-window beta to changes in the tradable 10-year US yield series,
less its longer-window beta.  It captures changing rate-transmission exposure;
inputs through 2035-01-17 only, with forward returns used solely for evaluation.
"""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-01-17')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""rate_signal=p['US10Y'].pct_change(fill_method=None)
f=beta(r,rate_signal,20,mp=15)-beta(r,rate_signal,60,mp=45)
print('CANDIDATE continuous_us10y_beta_migration_20_60obs')"""
assert old in s
s=s.replace(old,new)
# Keep the DXY variable within the library comparator definitions unchanged.
exec(compile(s,'miner_1_continuous_us10y_beta_migration','exec'))

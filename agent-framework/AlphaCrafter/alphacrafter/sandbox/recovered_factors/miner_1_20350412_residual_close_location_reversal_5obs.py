"""Candidate validation: residual close-location reversal (5 observations), full active-library audit."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-04-11')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Short-horizon idiosyncratic close-location reversal: persistent closes near the
# daily extreme after residual moves are treated as exhaustion, not trend.
# Magnitude weighting excludes mechanically quiet sessions; negate for reversal.
pressure=(clv-.5)*res.abs().div(v.replace(0,np.nan)).clip(upper=4)
f=-pressure.rolling(5,min_periods=4).mean()
print('CANDIDATE residual_close_location_reversal_5obs')"""
assert old in s
s=s.replace(old,new)
exec(compile(s,'miner_1_residual_close_location_reversal_5obs','exec'))

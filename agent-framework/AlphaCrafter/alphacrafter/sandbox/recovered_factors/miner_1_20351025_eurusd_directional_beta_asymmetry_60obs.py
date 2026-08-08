"""Validate one candidate: EURUSD directional beta asymmetry (60 observations)."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-10-24')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Difference in asset sensitivity to EURUSD appreciation versus depreciation.
# A positive value means the asset has had a stronger beta on EUR-strength days,
# capturing directional FX-risk transmission rather than unconditional FX beta.
eur=macro('EURUSD')
f=beta(r,eur,60,eur>0,mp=18)-beta(r,eur,60,eur<0,mp=18)
print('CANDIDATE eurusd_directional_beta_asymmetry_60obs')"""
assert old in s
s=s.replace(old,new)
s=s.replace("if H==10:","if H==5:")
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)","('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'eurusd_directional_beta_asymmetry_60obs','exec'))

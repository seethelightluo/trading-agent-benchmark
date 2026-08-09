"""Single-candidate validation: severity-weighted residual-downside close-location improvement (20 vs 60)."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-07-18')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# On idiosyncratically weak sessions, weight close location by residual-loss severity.
# A rising weighted close location means the largest idiosyncratic selloffs are increasingly absorbed intraday.
sev=(-res).clip(lower=0)
num20=clv.mul(sev).rolling(20,min_periods=8).sum(); den20=sev.rolling(20,min_periods=8).sum()
num60=clv.mul(sev).rolling(60,min_periods=20).sum(); den60=sev.rolling(60,min_periods=20).sum()
f=num20/den20-num60/den60
print('CANDIDATE severity_weighted_residual_downside_close_location_improvement_20_60obs')"""
assert old in s
s=s.replace(old,new)
s=s.replace("if H==10:","if H==20:")
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)","('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'severity_weighted_residual_downside_close_location_improvement','exec'))

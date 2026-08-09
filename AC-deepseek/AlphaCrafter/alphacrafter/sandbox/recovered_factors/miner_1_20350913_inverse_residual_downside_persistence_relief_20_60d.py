"""Validate one candidate: residual downside persistence versus recovery asymmetry (20/60)."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-09-12')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Difference in recent versus baseline idiosyncratic downside persistence. Residual losses
# are converted to a cross-asset-relative, volatility-normalized severity indicator so this
# signal favors assets whose relative downside episodes have recently become less persistent.
sev=(-res.div(v.replace(0,np.nan))).clip(lower=0,upper=5)
f=-(sev.rolling(20,min_periods=15).mean()-sev.rolling(60,min_periods=45).mean())
print('CANDIDATE inverse_residual_downside_persistence_relief_20_60d')"""
assert old in s
s=s.replace(old,new)
s=s.replace("if H==10:","if H==5:")
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)","('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'inverse_residual_downside_persistence_relief_20_60d','exec'))

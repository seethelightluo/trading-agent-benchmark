"""Single-candidate audit: dispersion-conditioned idiosyncratic reversal."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-08-15')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Reversal is activated only when cross-asset dispersion is elevated: after a
# heterogeneous shock, a negative 5-day idiosyncratic move is more likely to be
# an asset-specific overshoot than a common risk-on/off trend.
disp=r.std(axis=1)
high_disp=disp>disp.rolling(60,min_periods=45).median()
idvol=res.rolling(20,min_periods=15).std()
f=(-res.rolling(5,min_periods=4).sum()/(idvol*np.sqrt(5))).where(high_disp,axis=0)
print('CANDIDATE dispersion_conditioned_idiosyncratic_volnorm_reversal_5_20obs')"""
assert old in s
s=s.replace(old,new)
s=s.replace("if H==10:","if H==5:")
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)","('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'dispersion_conditioned_idiosyncratic_volnorm_reversal_5_20obs','exec'))

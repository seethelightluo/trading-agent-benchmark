"""Validate one idea: dispersion-conditioned volume participation response."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-11-07')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Asset-specific abnormal trading participation, emphasized when the prior
# cross-asset dispersion is elevated. High volume in a dispersion regime
# can reveal informed commitment rather than an unconditional liquidity level.
disp=r.std(axis=1); disp_z=(disp-disp.rolling(60,min_periods=45).mean())/disp.rolling(60,min_periods=45).std()
volsur=np.log(vo.rolling(5,min_periods=4).mean()/vo.rolling(60,min_periods=40).mean())
f=volsur.mul(disp_z.shift(1).clip(lower=0,upper=3),axis=0)
print('CANDIDATE dispersion_conditioned_volume_participation_response_5_60obs')"""
assert old in s
s=s.replace(old,new)
s=s.replace("if H==10:","if H==5:")
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)","('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'dispersion_conditioned_volume_participation_response_5_60obs','exec'))

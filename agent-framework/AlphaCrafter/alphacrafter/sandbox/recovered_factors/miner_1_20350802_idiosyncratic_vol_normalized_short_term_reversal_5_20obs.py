"""Single-candidate validation: idiosyncratic short-term reversal, residual-return / idiosyncratic-vol normalized."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-08-01')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Negative recent idiosyncratic performance relative to its own residual risk
# is hypothesized to mean-revert.  Median-market residualization avoids merely
# ranking common cross-asset risk-on/risk-off moves.
idvol=res.rolling(20,min_periods=15).std()
f=-res.rolling(5,min_periods=4).sum()/(idvol*np.sqrt(5))
print('CANDIDATE idiosyncratic_vol_normalized_short_term_reversal_5_20obs')"""
assert old in s
s=s.replace(old,new)
s=s.replace("if H==10:","if H==5:")
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)","('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'idiosyncratic_vol_normalized_short_term_reversal_5_20obs','exec'))

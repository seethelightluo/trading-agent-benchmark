"""Validate one candidate: residual downside-event age (20 day capped)."""
p='scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2035-10-10')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""# Age of the latest idiosyncratic downside shock. A shock is residual return
# below one own 20d residual-volatility unit; higher values mean a longer uninterrupted
# residual recovery/no-new-shock interval. Cap at 20 to keep interpretation local.
event=res.lt(-v)
f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
    last=np.where(event[a].fillna(False).to_numpy(), np.arange(len(p)), np.nan)
    last=pd.Series(last,index=p.index).ffill()
    f[a]=(pd.Series(np.arange(len(p)),index=p.index)-last).clip(upper=20)
f=f.where(v.notna())
print('CANDIDATE residual_downside_event_age_20d')"""
assert old in s
s=s.replace(old,new)
s=s.replace("if H==10:","if H==5:")
s=s.replace("('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_34','2030-01-01',END)","('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('2033_35','2033-01-01',END)")
exec(compile(s,'residual_downside_event_age_20d','exec'))

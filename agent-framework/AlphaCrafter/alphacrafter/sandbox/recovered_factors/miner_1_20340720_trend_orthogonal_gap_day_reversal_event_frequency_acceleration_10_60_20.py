"""Miner 1: trend-orthogonal gap/day reversal-event frequency acceleration."""
src=open('scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py').read()
src=src.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2034-07-19')")
old="f=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=45).mean()"
new="""gap=op/p.shift(1)-1; day=p/op-1
valid=(gap.notna() & day.notna() & (gap!=0) & (day!=0))
event=((gap*day)<0).astype(float).where(valid)
raw=event.rolling(10,min_periods=8).mean()-event.rolling(60,min_periods=45).mean()
trend=(p/p.shift(20)-1)/v
f=orth(raw,trend)
print('FACTOR trend_orthogonal_gap_day_reversal_event_frequency_acceleration_10_60_20',flush=True)"""
assert old in src
src=src.replace(old,new).replace("mx=-1\nfor n,s in L.items():", "mx=-1;who='NONE';cells=0\nfor n,s in L.items():")
src=src.replace("'min_n',min(ns)","'min_n',(min(ns) if ns else 0)")
src=src.replace("round(z.mean()/z.std(ddof=1),6)","round(z.mean()/z.std(ddof=1),6) if len(z)>1 else np.nan")
src=src.replace("round((z>0).mean(),5)","round((z>0).mean(),5) if len(z) else np.nan")
src=src.replace("round(zz.mean()/zz.std(ddof=1),6)","round(zz.mean()/zz.std(ddof=1),6) if len(zz)>1 else np.nan")
src=src.replace("round((zz>0).mean(),5)","round((zz>0).mean(),5) if len(zz) else np.nan")
src=src.replace('continuous close-location pressure.', 'trend-orthogonal gap/day reversal-event frequency acceleration.')
exec(src)

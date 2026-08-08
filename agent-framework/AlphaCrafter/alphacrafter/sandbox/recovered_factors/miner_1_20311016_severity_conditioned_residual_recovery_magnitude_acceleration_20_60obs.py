"""One idea: severity-conditioned residual recovery magnitude acceleration; completed-bar 2031-10-15 validation."""
from pathlib import Path
src=Path('scripts/miner_1_20311002_revalidate_residual_downside_event_spacing_relief_5_60obs.py').read_text()
src=src.replace('quarterly revalidation of residual-downside event-spacing relief; 2031-10-01 completed-bar cutoff','severity-conditioned residual recovery magnitude acceleration; 2031-10-15 completed-bar cutoff').replace("END=pd.Timestamp('2031-10-01')", "END=pd.Timestamp('2031-10-15')")
old="""neg_res=res.shift(1)<0
# Higher value: a continuous recent increase in the spacing between idiosyncratic down days relative to 60d normal spacing.
gap=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
 last=np.nan; out=[]
 for event in neg_res[a].fillna(False):
  if event: last=0.
  elif np.isfinite(last): last+=1
  out.append(last)
 gap[a]=out
f=np.log((gap.rolling(5,min_periods=4).mean()+1)/(gap.rolling(60,min_periods=45).mean()+1))"""
new="""neg_res=res.shift(1)<(-0.5*res.rolling(60,min_periods=45).std())
# Higher value: next-session residual rebounds after materially negative residual days have strengthened over 20 observations versus 60 observations.
recovery=res.where(neg_res)
f=recovery.rolling(20,min_periods=4).mean()-recovery.rolling(60,min_periods=12).mean()"""
assert old in src
src=src.replace(old,new).replace('residual_downside_event_spacing_relief_5_60obs','severity_conditioned_residual_recovery_magnitude_acceleration_20_60obs')
src=src.replace("mx=-1\nfor n,x", "mx=-1;who='NO_VALID_LIBRARY_PAIR';cells=0\nfor n,x")
exec(compile(src,'severity_recovery_candidate.py','exec'))

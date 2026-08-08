"""One idea: continuous severity-weighted residual recovery slope; completed-bar 2031-10-29 validation."""
from pathlib import Path
src=Path('scripts/miner_1_20311002_revalidate_residual_downside_event_spacing_relief_5_60obs.py').read_text()
src=src.replace('quarterly revalidation of residual-downside event-spacing relief; 2031-10-01 completed-bar cutoff','continuous severity-weighted residual recovery slope; 2031-10-29 completed-bar cutoff').replace("END=pd.Timestamp('2031-10-01')", "END=pd.Timestamp('2031-10-29')")
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
new="""# At each date, measure the residual rebound following the prior day's idiosyncratic loss,
# weighted continuously by that loss's trailing-volatility-normalized severity.  Recent
# 20-observation recovery strength relative to the 60-observation baseline is the signal.
sev=(-res.shift(1)/res.rolling(60,min_periods=45).std()).clip(lower=0,upper=4)
weighted_recovery=res*sev
f=weighted_recovery.rolling(20,min_periods=15).mean()-weighted_recovery.rolling(60,min_periods=45).mean()"""
assert old in src
src=src.replace(old,new).replace('residual_downside_event_spacing_relief_5_60obs','continuous_severity_weighted_residual_recovery_slope_20_60obs')
src=src.replace("mx=-1\nfor n,x", "mx=-1;who='NO_VALID_LIBRARY_PAIR';cells=0\nfor n,x")
exec(compile(src,'continuous_severity_recovery_slope_candidate.py','exec'))

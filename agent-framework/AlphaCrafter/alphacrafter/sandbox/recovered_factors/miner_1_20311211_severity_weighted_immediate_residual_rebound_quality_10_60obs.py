"""One idea: severity-weighted immediate residual rebound quality, validated through completed bar 2031-12-10."""
from pathlib import Path
src=Path('scripts/miner_1_20311002_revalidate_residual_downside_event_spacing_relief_5_60obs.py').read_text()
src=src.replace('quarterly revalidation of residual-downside event-spacing relief; 2031-10-01 completed-bar cutoff','severity-weighted immediate residual rebound quality; 2031-12-10 completed-bar cutoff')
src=src.replace("END=pd.Timestamp('2031-10-01')", "END=pd.Timestamp('2031-12-10')")
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
new="""# An immediate idiosyncratic rebound earns a high score only when it follows a severe
# prior-day idiosyncratic loss.  The 10-observation severity-weighted mean is divided
# by its 60-observation counterpart, isolating recovery quality from an asset's level
# of typical residual volatility.
prior_severity=(-res.shift(1)/res.rolling(60,min_periods=45).std()).clip(lower=0,upper=4)
immediate_rebound=res*prior_severity
recent=immediate_rebound.rolling(10,min_periods=7).mean()
baseline=immediate_rebound.rolling(60,min_periods=45).mean()
f=recent-baseline"""
assert old in src
src=src.replace(old,new).replace('residual_downside_event_spacing_relief_5_60obs','severity_weighted_immediate_residual_rebound_quality_10_60obs')
exec(compile(src,'severity_weighted_immediate_residual_rebound_quality_10_60obs.py','exec'))

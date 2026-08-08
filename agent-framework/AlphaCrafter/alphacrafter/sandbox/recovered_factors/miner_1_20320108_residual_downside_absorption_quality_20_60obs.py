"""One idea: residual-downside absorption quality; completed-bar cutoff 2032-01-07."""
from pathlib import Path
src=Path('scripts/miner_1_20311002_revalidate_residual_downside_event_spacing_relief_5_60obs.py').read_text()
src=src.replace('quarterly revalidation of residual-downside event-spacing relief; 2031-10-01 completed-bar cutoff','residual-downside absorption quality; 2032-01-07 completed-bar cutoff')
src=src.replace("END=pd.Timestamp('2031-10-01')", "END=pd.Timestamp('2032-01-07')")
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
new="""# A severe idiosyncratic down session that closes high in its own intraday range
# reflects absorption rather than persistent selling. Compare severity-weighted
# absorption over 20 sessions with an asset's 60-session normal level.
clv0=((p-lo)/(hi-lo).replace(0,np.nan)).clip(0,1)
severity=(-res/res.rolling(60,min_periods=45).std()).clip(lower=0,upper=4)
absorption=severity*clv0
recent=absorption.rolling(20,min_periods=15).sum()/(severity.rolling(20,min_periods=15).sum()+1e-12)
baseline=absorption.rolling(60,min_periods=45).sum()/(severity.rolling(60,min_periods=45).sum()+1e-12)
f=recent-baseline"""
assert old in src
src=src.replace(old,new).replace('residual_downside_event_spacing_relief_5_60obs','residual_downside_absorption_quality_20_60obs')
exec(compile(src,'residual_downside_absorption_quality_20_60obs.py','exec'))

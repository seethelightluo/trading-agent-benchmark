"""One interpretable idea: residual-downside post-event recovery persistence; completed-bar validation."""
from pathlib import Path
src=Path('scripts/miner_1_20310724_residual_downside_event_spacing_relief_5_60obs.py').read_text()
src=src.replace('"""One idea: residual-downside event-spacing relief; 2031-07-24 completed-bar validation."""','"""One idea: residual-downside post-event recovery persistence; completed-bar validation."""')
src=src.replace("END=pd.Timestamp('2031-07-23')","END=pd.Timestamp('2031-09-17')")
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
new="""neg_res=res.shift(1)<0
# Higher value: after an idiosyncratic downside event, next-session residual recovery has become more persistent versus the asset's baseline.
# Outcome at t is known only if the triggering downside event occurred at t-1.
recover=(res>0).where(neg_res)
recent=recover.rolling(20,min_periods=5).mean()
baseline=recover.rolling(60,min_periods=15).mean()
f=recent-baseline"""
assert old in src
src=src.replace(old,new).replace('residual_downside_event_spacing_relief_5_60obs','residual_downside_postevent_recovery_persistence_20_60obs')
# Correct the library's inverse transition construction that uses same-day event/outcome incompatibly is retained only as comparator
Path('scripts/miner_1_20310918_residual_downside_postevent_recovery_persistence_20_60obs.py').write_text(src)
exec(compile(src,'candidate_generated.py','exec'))

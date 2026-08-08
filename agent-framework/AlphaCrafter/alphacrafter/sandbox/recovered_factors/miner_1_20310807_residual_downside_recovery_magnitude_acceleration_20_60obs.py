"""One idea: residual downside recovery magnitude acceleration; 2031-08-07 completed-bar validation."""
from pathlib import Path
src=Path('scripts/miner_1_20310724_residual_downside_event_spacing_relief_5_60obs.py').read_text()
src=src.replace('"""One idea: residual-downside event-spacing relief; 2031-07-24 completed-bar validation."""','"""One idea: residual downside recovery magnitude acceleration; completed-bar validation."""').replace("END=pd.Timestamp('2031-07-23')","END=pd.Timestamp('2031-08-06')")
old="neg_res=res.shift(1)<0\n# Higher value: a continuous recent increase in the spacing between idiosyncratic down days relative to 60d normal spacing.\ngap=pd.DataFrame(index=p.index,columns=A,dtype=float)\nfor a in A:\n last=np.nan; out=[]\n for event in neg_res[a].fillna(False):\n  if event: last=0.\n  elif np.isfinite(last): last+=1\n  out.append(last)\n gap[a]=out\nf=np.log((gap.rolling(5,min_periods=4).mean()+1)/(gap.rolling(60,min_periods=45).mean()+1))"
new="neg_res=res.shift(1)<0\n# Higher value: residual rebounds immediately after idiosyncratic downside days have recently strengthened versus the asset's 60-observation baseline.\nrecovery=res.where(neg_res)\nf=recovery.rolling(20,min_periods=8).mean()-recovery.rolling(60,min_periods=25).mean()"
assert old in src
src=src.replace(old,new).replace('residual_downside_event_spacing_relief_5_60obs','residual_downside_recovery_magnitude_acceleration_20_60obs').replace("'SELECTED'","'SELECTED'")
exec(compile(src,'candidate_generated.py','exec'))

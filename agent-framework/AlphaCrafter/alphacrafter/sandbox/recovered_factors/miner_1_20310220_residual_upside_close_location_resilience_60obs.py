"""Single idea: residual-upside close-location resilience, 60 observations.
Higher values mean the asset closes nearer its daily high following its own
idiosyncratic (market-residual) positive day; this is a conditional continuation-quality signal."""
from pathlib import Path
src=Path('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2030-08-21')", "END=pd.Timestamp('2031-02-19')")
src=src.replace('common_stress_relative_volume_participation_response_60obs','residual_upside_close_location_resilience_60obs')
src=src.replace('common-stress relative-volume participation response','residual-upside close-location resilience')
src=src.replace('abnormal log volume sensitivity to lagged common cross-asset stress','mean close location conditional on prior-day positive market-residual return')
old="particip=np.log(V/V.rolling(20,min_periods=15).median())\nf=pd.DataFrame({a:particip[a].rolling(60,min_periods=30).cov(stress).div(stress.rolling(60,min_periods=30).var()) for a in A})"
new="""loc0=(p-lo).div((hi-lo).replace(0,np.nan))
basebeta=pd.DataFrame({a:r[a].rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var()) for a in A})
eps=r-basebeta.mul(m,axis=0)
# Conditional mean requires at least 12 qualifying observations in the last 60.
f=pd.DataFrame({a:loc0[a].where(eps[a].shift()>0).rolling(60,min_periods=12).mean() for a in A})"""
assert old in src
src=src.replace(old,new)
src=src.replace('log(volume[t]/median(volume,20)[t])','mean_60(close_location[t] | residual_return[t-1] > 0)')
src=src.replace("f.to_pickle('scripts/miner_1_20300822_residual_upside_close_location_resilience_60obs_candidate_signal.pkl')", "f.to_pickle('scripts/miner_1_20310220_residual_upside_close_location_resilience_60obs_candidate_signal.pkl')")
exec(compile(src,'residual_upside_location_harness','exec'))

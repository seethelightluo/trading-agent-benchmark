"""Single idea: common-stress relative-range expansion response, 60 observations."""
# Reuse the fully audited validation harness, substituting one interpretable candidate
# and adding the last admitted candidate to the exact library comparison set.
from pathlib import Path
src=Path('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2030-08-21')", "END=pd.Timestamp('2030-09-04')")
src=src.replace('common_stress_relative_volume_participation_response_60obs','common_stress_relative_range_expansion_response_60obs')
src=src.replace('common-stress relative-volume participation response','common-stress relative-range expansion response')
src=src.replace('abnormal log volume sensitivity to lagged common cross-asset stress','abnormal log intraday-range sensitivity to lagged common cross-asset stress')
src=src.replace("particip=np.log(V/V.rolling(20,min_periods=15).median())", "particip=np.log(((hi-lo).abs()/p) / (((hi-lo).abs()/p).rolling(20,min_periods=15).median()))")
src=src.replace("log(volume[t]/median(volume,20)[t])", "log((high[t]-low[t])/close[t] / median_20((high-low)/close)[t])")
# Add the previous admission as a mandatory 27th comparison; align its saved signal to cutoff.
src=src.replace("lib={'realized_volatility_20obs':v,", "lib={'common_stress_relative_volume_participation_response_60obs':pd.read_pickle('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs_candidate_signal.pkl').loc[:END], 'realized_volatility_20obs':v,")
exec(compile(src, 'range_response_harness', 'exec'))

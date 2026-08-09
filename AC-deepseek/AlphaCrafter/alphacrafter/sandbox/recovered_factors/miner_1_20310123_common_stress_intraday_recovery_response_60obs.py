"""Single idea: common-stress intraday-recovery response, 60 observations.
Higher signal means an asset's open-to-close return has historically been more
positive after lagged broad cross-asset stress.  It is deliberately distinct
from prior overnight-gap and close-location responses."""
from pathlib import Path
src=Path('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2030-08-21')", "END=pd.Timestamp('2031-01-22')")
src=src.replace('common_stress_relative_volume_participation_response_60obs','common_stress_intraday_recovery_response_60obs')
src=src.replace('common-stress relative-volume participation response','common-stress intraday-recovery response')
src=src.replace('abnormal log volume sensitivity to lagged common cross-asset stress','open-to-close return sensitivity to lagged common cross-asset stress')
src=src.replace("particip=np.log(V/V.rolling(20,min_periods=15).median())", "op=pd.DataFrame({a:ld(a,'open') for a in A}); particip=np.log(p/op).clip(-.25,.25)")
src=src.replace("log(volume[t]/median(volume,20)[t])", "log(close[t]/open[t])")
src=src.replace("f.to_pickle('scripts/miner_1_20300822_common_stress_intraday_recovery_response_60obs_candidate_signal.pkl')", "f.to_pickle('scripts/miner_1_20310123_common_stress_intraday_recovery_response_60obs_candidate_signal.pkl')")
exec(compile(src,'intraday_recovery_harness','exec'))

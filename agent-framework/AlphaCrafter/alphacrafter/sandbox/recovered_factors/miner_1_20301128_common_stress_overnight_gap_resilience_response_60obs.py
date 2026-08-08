"""Single idea: common-stress overnight-gap resilience response, 60 observations."""
from pathlib import Path
# Audited cross-asset IC harness and reconstructed contemporaneous library signals.
src=Path('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2030-08-21')", "END=pd.Timestamp('2030-11-27')")
src=src.replace('common_stress_relative_volume_participation_response_60obs','common_stress_overnight_gap_resilience_response_60obs')
src=src.replace('common-stress relative-volume participation response','common-stress overnight-gap resilience response')
src=src.replace('abnormal log volume sensitivity to lagged common cross-asset stress','overnight gap-return sensitivity to lagged common cross-asset stress')
# The gap is known at the open of t; stress only uses completed t-1 returns. Positive beta means gaps are relatively resilient after broad stress.
src=src.replace("particip=np.log(V/V.rolling(20,min_periods=15).median())", "op=pd.DataFrame({a:ld(a,'open') for a in A}); particip=np.log(op/p.shift()).clip(-.25,.25)")
src=src.replace("log(volume[t]/median(volume,20)[t])", "log(open[t]/close[t-1])")
# Mandatory comparisons to all reconstructed factors plus signal artifacts for the two newest admitted factors.
old="lib={'realized_volatility_20obs':v,"
new="lib={'residualized_volnorm_drawdown_repair_speed_20_60obs':pd.read_pickle('scripts/miner_1_20301017_residualized_volnorm_drawdown_repair_speed_20_60obs_candidate_signal.pkl').loc[:END], 'inverse_continuous_dispersion_weighted_residual_persistence_60obs':pd.read_pickle('scripts/miner_3_20301031_inverse_continuous_dispersion_weighted_residual_persistence_60obs_candidate_signal.pkl').loc[:END], 'realized_volatility_20obs':v,"
src=src.replace(old,new)
# Do not overwrite an admitted factor's historical candidate artifact.
src=src.replace("f.to_pickle('scripts/miner_1_20300822_common_stress_overnight_gap_resilience_response_60obs_candidate_signal.pkl')", "f.to_pickle('scripts/miner_1_20301128_common_stress_overnight_gap_resilience_response_60obs_candidate_signal.pkl')")
exec(compile(src,'gap_resilience_harness','exec'))

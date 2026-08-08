"""Single idea: volatility-normalized drawdown repair speed, 60 observations."""
from pathlib import Path
src=Path('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2030-08-21')", "END=pd.Timestamp('2030-10-02')")
old="""# Interpretable response: each asset's abnormal log volume sensitivity to lagged common cross-asset stress.
particip=np.log(V/V.rolling(20,min_periods=15).median())
f=pd.DataFrame({a:particip[a].rolling(60,min_periods=30).cov(stress).div(stress.rolling(60,min_periods=30).var()) for a in A})"""
new="""# Repair speed: recent five-day rebound from a rolling 20-day trough, normalized by
# own 20-day realized volatility; only meaningful while the asset remains below its 60-day peak.
vol20=r.rolling(20,min_periods=15).std()
trough=p.rolling(20,min_periods=15).min()
repair=np.log(p/trough.shift(5)).div(vol20*np.sqrt(5))
underwater=p.div(p.rolling(60,min_periods=45).max()).sub(1)
f=repair.where(underwater<0)"""
assert old in src
src=src.replace(old,new)
src=src.replace('common_stress_relative_volume_participation_response_60obs','volnorm_drawdown_repair_speed_20_60obs')
src=src.replace('common-stress relative-volume participation response','volatility-normalized drawdown repair speed')
src=src.replace('abnormal log volume sensitivity to lagged common cross-asset stress','five-day trough rebound divided by 20-day volatility, while below 60-day peak')
src=src.replace("print('expression=rolling_60_beta(log(volume[t]/median(volume,20)[t]), -median_crossasset_return[t-1]/rolling_std_60(median_return)[t-1])')", "print('expression=log(close[t]/rolling_min(close,20)[t-5])/(rolling_std(return,20)[t]*sqrt(5)), conditioned on close[t]<rolling_max(close,60)[t]')")
# Restore the admitted volume-response factor in library dictionary after broad candidate renaming.
src=src.replace("lib={'realized_volatility_20obs':v,", "lib={'common_stress_relative_volume_participation_response_60obs':pd.read_pickle('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs_candidate_signal.pkl').loc[:END], 'realized_volatility_20obs':v,")
# Include exact candidate signal artifacts where available for additional admitted signals absent from base harness.
needle="mx=-1;missing=[]"
insert="""# Exact persisted signal artifacts for admitted factors not reconstructed above.
for nm,fn in {
'idiosyncratic_upside_tail_skewness_60obs':'scripts/miner_2_candidate_signal.pkl',
'residual_downside_close_location_recovery_60obs':'scripts/miner_3_residualized_trough_recovery_slope_candidate_signal.pkl',
}.items():
 try: lib[nm]=pd.read_pickle(fn).loc[:END]
 except Exception as e: print('ARTIFACT_UNAVAILABLE',nm,type(e).__name__)
"""
assert needle in src
src=src.replace(needle,insert+needle)
src=src.replace("f.to_pickle('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs_candidate_signal.pkl')", "f.to_pickle('scripts/miner_1_20301003_volnorm_drawdown_repair_speed_20_60obs_candidate_signal.pkl')")
Path('scripts/miner_1_20301003_volnorm_drawdown_repair_speed_20_60obs.py').write_text(src)
exec(compile(src,'drawdown_repair_harness','exec'))

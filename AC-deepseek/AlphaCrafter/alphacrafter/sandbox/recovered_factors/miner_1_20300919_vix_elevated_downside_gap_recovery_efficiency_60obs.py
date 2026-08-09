"""Single idea: VIX-elevated downside-gap recovery efficiency, 60 observations."""
from pathlib import Path
src=Path('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2030-08-21')", "END=pd.Timestamp('2030-09-18')")
src=src.replace('common_stress_relative_volume_participation_response_60obs','vix_elevated_downside_gap_recovery_efficiency_60obs')
src=src.replace('common-stress relative-volume participation response','VIX-elevated downside-gap recovery efficiency')
src=src.replace('abnormal log volume sensitivity to lagged common cross-asset stress','mean intraday recovery after asset downside gaps on elevated VIX days')
# Open-to-close recovery is evaluated only after a negative overnight gap and lagged elevated macro volatility.
src=src.replace("p=pd.DataFrame({a:ld(a) for a in A});r=p.pct_change(fill_method=None);hi=", "p=pd.DataFrame({a:ld(a) for a in A});op=pd.DataFrame({a:ld(a,'open') for a in A});r=p.pct_change(fill_method=None);hi=")
src=src.replace("# Interpretable response: each asset's abnormal log volume sensitivity to lagged common cross-asset stress.\nparticip=np.log(V/V.rolling(20,min_periods=15).median())\nf=pd.DataFrame({a:particip[a].rolling(60,min_periods=30).cov(stress).div(stress.rolling(60,min_periods=30).var()) for a in A})", "# A downside gap is open below yesterday close. On elevated *lagged* VIX days, measure the same-session open-to-close recovery; rolling average is asset-specific resilience.\nvix_level=ld('VIX',idx=True); elevated=vix_level.shift(1)>vix_level.shift(1).rolling(60,min_periods=30).median()\ngap=op.div(p.shift(1)).sub(1); recovery=p.div(op).sub(1)\nf=pd.DataFrame({a:recovery[a].where((gap[a]<0)&elevated).rolling(60,min_periods=12).mean() for a in A})")
src=src.replace("print('expression=rolling_60_beta(log(volume[t]/median(volume,20)[t]), -median_crossasset_return[t-1]/rolling_std_60(median_return)[t-1])')", "print('expression=mean_60(close[t]/open[t]-1 | open[t]/close[t-1]-1 < 0 AND VIX[t-1] > median_60(VIX)[t-1])')")
# The existing admitted volume-stress signal is the 27th mandatory comparison.
src=src.replace("lib={'realized_volatility_20obs':v,", "lib={'common_stress_relative_volume_participation_response_60obs':pd.read_pickle('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs_candidate_signal.pkl').loc[:END], 'realized_volatility_20obs':v,")
exec(compile(src, 'gap_recovery_harness', 'exec'))

"""Single idea: cross-sectionally residualized common-stress intraday recovery response.
Removes each day's linear exposure of intraday recovery-stress beta to the
admitted close-location-stress beta, retaining the distinct recovery component."""
from pathlib import Path
src=Path('scripts/miner_1_20300822_common_stress_relative_volume_participation_response_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2030-08-21')", "END=pd.Timestamp('2031-02-05')")
src=src.replace('common_stress_relative_volume_participation_response_60obs','residualized_common_stress_intraday_recovery_response_60obs')
src=src.replace('common-stress relative-volume participation response','residualized common-stress intraday-recovery response')
src=src.replace('abnormal log volume sensitivity to lagged common cross-asset stress','cross-sectionally residualized open-to-close-return sensitivity to lagged common cross-asset stress')
src=src.replace("particip=np.log(V/V.rolling(20,min_periods=15).median())", "op=pd.DataFrame({a:ld(a,'open') for a in A}); particip=np.log(p/op).clip(-.25,.25)")
src=src.replace("f=pd.DataFrame({a:particip[a].rolling(60,min_periods=30).cov(stress).div(stress.rolling(60,min_periods=30).var()) for a in A})", "raw=pd.DataFrame({a:particip[a].rolling(60,min_periods=30).cov(stress).div(stress.rolling(60,min_periods=30).var()) for a in A}); loc0=(p-lo).div((hi-lo).replace(0,np.nan)); location=pd.DataFrame({a:loc0[a].rolling(60,min_periods=30).cov(stress).div(stress.rolling(60,min_periods=30).var()) for a in A}); f=raw.copy()*np.nan\nfor t in raw.index:\n q=pd.concat([raw.loc[t],location.loc[t]],axis=1).dropna()\n if len(q)>=8:\n  coef=np.polyfit(q.iloc[:,1],q.iloc[:,0],1); f.loc[t]=raw.loc[t]-(coef[1]+coef[0]*location.loc[t])")
src=src.replace("log(volume[t]/median(volume,20)[t])", "resid_cs( beta_60(log(close[t]/open[t]), stress[t]), beta_60(close_location[t], stress[t]) )")
src=src.replace("f.to_pickle('scripts/miner_1_20300822_residualized_common_stress_intraday_recovery_response_60obs_candidate_signal.pkl')", "f.to_pickle('scripts/miner_1_20310206_residualized_common_stress_intraday_recovery_response_60obs_candidate_signal.pkl')")
exec(compile(src,'residualized_intraday_recovery_harness','exec'))

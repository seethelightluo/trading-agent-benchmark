import json
p='factors/miner_2_20311030_continuous_vix_scaled_inflation_impulse_residual_loading_contraction_60_20d.json'
d=json.load(open(p))
m=d['validation']['metrics']
m.update({'daily_paper_ic':0.074740,'daily_paper_icir':0.223733,'ic_std':0.334058,'ic_standard_error':0.008640,'ic_hit_ratio':0.593311,'ic_dates':1495,'mean_valid_instruments':14.993311,'universe_instruments':15,'coverage':0.462254,'valid_factor_cells':27444,'mean_daily_rank_turnover':0.121186,'turnover_dates':1514,'driver_nonzero_fraction':0.380243,'max_abs_library_correlation':0.423333,'most_correlated_library_factor':'residual_positive_oil_change_shock_loading_contraction_20_60d','library_correlation_cells':22800,'library_factors_screened':34,'decay':{'1d':{'ic':0.006572,'icir':0.020336,'ic_dates':1514},'5d':{'ic':0.037709,'icir':0.114411,'ic_dates':1510},'10d':{'ic':0.056347,'icir':0.169524,'ic_dates':1505},'20d':{'ic':0.074740,'icir':0.223733,'ic_dates':1495}}})
d['version']='2032-07-22'
d['validation']['period']='2020-01-01 through 2032-07-21; completed-close visibility-safe cutoff'
d['validation']['regime_notes']='At 10 days: 2025-2026 has 66 IC dates, IC +0.032863, ICIR +0.106696, hit 0.515152; 2027 onward has 1,439 dates, IC +0.057424, ICIR +0.172177, hit 0.574010. No 2020-2024 observations survived rolling-history requirements. Full sample clears gates at 5d, 10d and 20d; all IC dates require >=8 assets and average 14.99. Revalidation through 2032-07-21 strengthens 20d IC/ICIR and confirms orthogonality below 0.5.'
d['validation']['timestamp']='2032-07-22T09:30:00'; d['last_validated']='2032-07-22';d['validation_timestamp']='2032-07-22'
d['benchmark_admission']['selected_metrics'].update({'ic':0.074740,'icir':0.223733,'max_abs_library_correlation':0.423333,'quality':0.01672170242})
open(p,'w').write(json.dumps(d,indent=2)+'\n')

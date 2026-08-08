import json
path='factors/miner_2_20311030_continuous_vix_scaled_inflation_impulse_residual_loading_contraction_60_20d.json'
d=json.load(open(path))
v=d['validation']; m=v['metrics']
v['period']='2020-01-01 through 2033-01-19; completed-close visibility-safe cutoff'
v['status']='EFFECTIVE'; v['timestamp']='2033-01-20T09:30:00'
d['last_validated']='2033-01-20'; d['validation_timestamp']='2033-01-20'; d['version']='2033-01-20'
m.update({'daily_paper_ic':0.066153,'daily_paper_icir':0.197447,'ic_std':0.335041,'ic_standard_error':0.008392,'ic_hit_ratio':0.579046,'ic_dates':1594,'mean_valid_instruments':14.993726,'universe_instruments':15,'coverage':0.479354,'valid_factor_cells':29394,'mean_daily_rank_turnover':0.123244,'turnover_dates':1644,'driver_nonzero_fraction':0.379403,'max_abs_library_correlation':0.413821,'most_correlated_library_factor':'residual_positive_oil_change_shock_loading_contraction_20_60d','library_correlation_cells':24750,'library_factors_screened':34,'decay':{'1d':{'ic':0.005466,'icir':0.016967,'ic_dates':1613},'5d':{'ic':0.031243,'icir':0.095352,'ic_dates':1609},'10d':{'ic':0.047138,'icir':0.14142,'ic_dates':1604},'20d':{'ic':0.066153,'icir':0.197447,'ic_dates':1594}}})
v['regime_notes']='At 10 days: 2025-2026 has 66 IC dates, IC +0.032863, ICIR +0.106696, hit 0.515152; 2027 onward has 1,538 dates, IC +0.047751, ICIR +0.142777, hit 0.561118. No 2020-2024 observations survived rolling-history requirements. Full sample clears shared admission gates at 5d, 10d and 20d. All IC dates require >=8 assets and average 14.99. Full-library independence remains satisfied: maximum absolute Spearman 0.413821 < 0.5.'
d['benchmark_admission']['selected_metrics'].update({'ic':0.066153,'icir':0.197447,'quality':0.013062,'max_abs_library_correlation':0.413821})
open(path,'w').write(json.dumps(d,indent=2)+'\n')

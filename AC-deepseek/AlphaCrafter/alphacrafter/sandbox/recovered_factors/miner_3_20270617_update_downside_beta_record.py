import json
p='factors/miner_3_20270325_downside_beta_asymmetry_30.json'
with open(p,encoding='utf-8') as f:d=json.load(f)
m=d['validation']['metrics']
m.update({'ic':0.066440,'icir':0.253583,'ic_horizon_days':10,'ic_dates':218,'ic_hit_ratio':0.6009,'mean_valid_instruments':14.89,'ic_standard_error':0.017745,'signal_cell_coverage':0.197463,'mean_daily_rank_turnover':0.097854,'concentration':'Cross-sectional beta-asymmetry ranks; mean 14.89 valid of 15 per IC date','decay':{'1d':{'ic':0.022307,'icir':0.078116,'dates':227},'5d':{'ic':0.021118,'icir':0.080688,'dates':223},'10d':{'ic':0.066440,'icir':0.253583,'dates':218},'20d':{'ic':0.063026,'icir':0.208987,'dates':208}},'max_abs_library_correlation':0.100812,'closest_library_factor':'miner_2_20270520_inverse_residual_return_skewness_20','common_signal_observations_closest':2706,'admission_quality_score':0.016849})
d['validation']['period']='aligned available panel through 2027-06-16; forward labels restricted to visible research panel'
d['validation']['status']='EFFECTIVE'
d['validation']['regime_notes']='All 218 ten-day IC observations remain in the available 2026-27 aligned segment (IC 0.066440, ICIR 0.253583, positive-IC hit ratio 60.09%). No earlier aligned IC observations are available in the accessible panel. The signal has decayed from its March validation but remains above shared admission thresholds, broadly covered, low-turnover, and diversified from the other 14 admitted signals; retain with conservative weight.'
d['last_validated']='2027-06-17T00:00:00Z';d['revalidation_due']='2027-09-17'
d['benchmark_admission']['selected_metrics']={'ic':0.066440,'icir':0.253583,'metric_path':'validation.metrics.decay.10d','max_abs_library_correlation':0.100812,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.016849};d['benchmark_admission']['revalidated_at']='2027-06-17T00:00:00Z'
with open(p,'w',encoding='utf-8') as f:json.dump(d,f,indent=2);f.write('\n')

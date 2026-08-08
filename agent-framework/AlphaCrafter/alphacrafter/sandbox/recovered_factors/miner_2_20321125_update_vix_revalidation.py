import json
p='factors/miner_2_20311030_continuous_vix_scaled_inflation_impulse_residual_loading_contraction_60_20d.json'
d=json.load(open(p,encoding='utf8'))
d['version']='2032-11-25'
d['validation']['period']='2020-01-01 through 2032-11-24; completed-close visibility-safe cutoff'
d['validation']['status']='EFFECTIVE'
d['validation']['selected_horizon_days']=20
d['validation']['metrics']={
 'daily_paper_ic':0.067770,'daily_paper_icir':0.201870,'ic_std':0.335708,'ic_standard_error':0.008516,'ic_hit_ratio':0.580438,'ic_dates':1554,'mean_valid_instruments':14.993565,'universe_instruments':15,'coverage':0.474209,'valid_factor_cells':28794,'mean_daily_rank_turnover':0.123661,'turnover_dates':1604,'driver_nonzero_fraction':0.376729,'max_abs_library_correlation':0.412695,'most_correlated_library_factor':'residual_positive_oil_change_shock_loading_contraction_20_60d','library_correlation_cells':24150,'library_factors_screened':34,
 'decay':{'1d':{'ic':0.006081,'icir':0.018840,'ic_dates':1573},'5d':{'ic':0.032361,'icir':0.098842,'ic_dates':1569},'10d':{'ic':0.048486,'icir':0.145436,'ic_dates':1564},'20d':{'ic':0.067770,'icir':0.201870,'ic_dates':1554}}}
d['validation']['regime_notes']='At 10 days: 2025-2026 has 66 IC dates, IC +0.032863, ICIR +0.106696, hit 0.515152; 2027 onward has 1,498 dates, IC +0.049174, ICIR +0.146992, hit 0.563418. No 2020-2024 observations survived rolling-history requirements. Full sample clears the shared gates at 5d, 10d and 20d. All IC dates require >=8 assets and average 14.99. Revalidation remains independent of the 34-signal admitted library (maximum absolute Spearman 0.412695 < 0.5).'
d['validation']['timestamp']='2032-11-25T09:30:00'
d['last_validated']='2032-11-25'; d['validation_timestamp']='2032-11-25'
d['benchmark_admission']['selected_metrics']={'ic':0.067770,'icir':0.201870,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.412695,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.0136817199}
json.dump(d,open(p,'w',encoding='utf8'),indent=2,ensure_ascii=False); open(p,'a').write('\n')
print('updated',p)

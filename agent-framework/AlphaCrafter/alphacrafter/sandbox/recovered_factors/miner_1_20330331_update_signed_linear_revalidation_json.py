import json
p='factors/miner_1_20330303_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d.json'
d=json.load(open(p,encoding='utf8'))
d['version']='2033-03-31 revalidation'
v=d['validation']
v['period']='2020-01-01 through 2033-03-30; visibility-safe completed-close cutoff'
v['timestamp']='2033-03-31T09:30:00'
v['status']='EFFECTIVE'
v['metrics'].update({
 'primary_horizon_days':20,'daily_paper_ic':0.058208,'daily_paper_icir':0.168130,
 'ic_std':0.346206,'ic_standard_error':0.008505,'ic_hit_ratio':0.566687,'ic_dates':1657,
 'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.012070,
 'signal_cell_coverage':0.346689,'valid_cells':21519,'mean_rank_turnover':0.209708,
 'turnover_dates':1676,'library_factors_screened':34,'max_abs_library_correlation':0.402100,
 'max_abs_library_correlation_factor':'miner_3_residual_upside_volume_confirmation_60d',
 'max_abs_library_correlation_common_cells':14798,
 'decay':{'1d':{'ic':0.015078,'icir':0.045816,'hit_ratio':0.508353,'ic_dates':1676},'5d':{'ic':0.034666,'icir':0.108082,'hit_ratio':0.542464,'ic_dates':1672},'10d':{'ic':0.042315,'icir':0.128977,'hit_ratio':0.547091,'ic_dates':1667},'20d':{'ic':0.058208,'icir':0.168130,'hit_ratio':0.566687,'ic_dates':1657}}
})
v['regime_notes']={'2020_2024_10d':{'ic_dates':0,'ic':None,'icir':None,'hit_ratio':None},'2025_2026_10d':{'ic_dates':48,'ic':0.009275,'icir':0.029474,'hit_ratio':0.541667},'2027_onward_10d':{'ic_dates':1619,'ic':0.043294,'icir':0.131789,'hit_ratio':0.547251},'interpretation':'Revalidation remains above the binding gates at 5, 10, and 20 days; strongest evidence is 20 days. The 2027-onward sample remains positive and stable, while the short 2025-26 partition remains weak. Mean IC cross-section is 10.01 assets, above the eight-instrument minimum.'}
d['last_validated']='2033-03-31T09:30:00'
d['validation_timestamp']='2033-03-31T09:30:00'
d['benchmark_admission']['selected_metrics'].update({'ic':0.058208,'icir':0.168130,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.402100,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.00978609264})
json.dump(d,open(p,'w',encoding='utf8'),indent=2,ensure_ascii=False)

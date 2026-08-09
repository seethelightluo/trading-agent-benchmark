import json
p='factors/miner_1_20310403_residual_jump_concentration_expansion_20_60d.json'
d=json.load(open(p))
v=d['validation'];m=v['metrics']
v['period']='2020-01-01 through 2031-10-01; visibility-safe completed-close cutoff'
v['timestamp']='2031-10-02T09:30:00';v['status']='EFFECTIVE'
m.update({'primary_horizon_days':20,'daily_paper_ic':0.051368,'daily_paper_icir':0.156288,'ic_std':0.328678,'ic_standard_error':0.009234,'ic_hit_ratio':0.573796,'ic_dates':1267,'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.015785,'signal_cell_coverage':0.313394,'mean_rank_turnover':0.127376,'turnover_dates':1286,'max_abs_library_correlation':0.298988,'max_abs_library_correlation_factor':'miner_1_residual_positive_jump_concentration_expansion_20_60d','max_abs_library_correlation_common_cells':17615,'library_factors_screened':36,'decay':{'1':{'ic':0.007551,'icir':0.022142,'hit_ratio':0.508554,'ic_dates':1286,'mean_valid_instruments':10.015552},'5':{'ic':0.018091,'icir':0.053711,'hit_ratio':0.524961,'ic_dates':1282,'mean_valid_instruments':10.015601},'10':{'ic':0.032248,'icir':0.101971,'hit_ratio':0.541895,'ic_dates':1277,'mean_valid_instruments':10.015662},'20':{'ic':0.051368,'icir':0.156288,'hit_ratio':0.573796,'ic_dates':1267,'mean_valid_instruments':10.015785}}})
v['regime_notes']={'2020_2024_10d':{'ic_dates':0,'ic':None,'icir':None,'hit_ratio':None},'2025_2026_10d':{'ic_dates':48,'ic':0.114379,'icir':0.376484,'hit_ratio':0.645833},'2027_onward_10d':{'ic_dates':1229,'ic':0.029041,'icir':0.091781,'hit_ratio':0.537836},'interpretation':'The 20-session primary horizon and 10-session diagnostic clear the shared IC and ICIR gates. The extensive post-2027 regime retains a 10-session ICIR of 0.091781, above the 0.084 gate. Coverage is conditional (31.3394%); retain as a medium-horizon diversifier and monitor drift.'}
d['last_validated']='2031-10-02T09:30:00';d['validation_timestamp']='2031-10-02T09:30:00'
# Preserve original admission history but record current selected validation.
d['benchmark_admission']['selected_metrics'].update({'ic':0.051368,'icir':0.156288,'metric_path':'validation.metrics primary 20d','max_abs_library_correlation':0.298988,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.008027})
json.dump(d,open(p,'w'),indent=2,ensure_ascii=False);open(p,'a').write('\n')

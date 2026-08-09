import json
path='factors/miner_1_20310417_residual_positive_jump_concentration_expansion_20_60d.json'
with open(path,encoding='utf8') as fh: d=json.load(fh)
m=d['validation']['metrics']
m.update({'primary_horizon_days':20,'daily_paper_ic':0.030611,'daily_paper_icir':0.093110,'ic_std':0.328764,'ic_standard_error':0.009386,'ic_hit_ratio':0.519967,'ic_dates':1227,'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.013040,'signal_cell_coverage':0.309511,'mean_rank_turnover':0.122999,'turnover_dates':1246,'max_abs_library_correlation':0.429992,'max_abs_library_correlation_factor':'miner_3_risk_adjusted_trend_20d','max_abs_library_correlation_common_cells':17209,'library_factors_screened':35,
'decay':{'1d':{'ic':0.011393,'icir':0.034648,'hit_ratio':0.519262,'ic_dates':1246,'mean_valid_instruments':10.012841},'5d':{'ic':0.000253,'icir':0.000767,'hit_ratio':0.495169,'ic_dates':1242,'mean_valid_instruments':10.012882},'10d':{'ic':-0.002275,'icir':-0.006958,'hit_ratio':0.491512,'ic_dates':1237,'mean_valid_instruments':10.012935},'20d':{'ic':0.030611,'icir':0.093110,'hit_ratio':0.519967,'ic_dates':1227,'mean_valid_instruments':10.013040}}})
d['validation']['period']='2020-01-01 through 2031-08-06; visibility-safe completed-close cutoff'
d['validation']['timestamp']='2031-08-07T09:30:00'
d['validation']['status']='EFFECTIVE'
d['validation']['regime_notes']={'2020_2024_10d':{'ic_dates':0,'ic':None,'icir':None,'hit_ratio':None},'2025_2026_10d':{'ic_dates':48,'ic':0.083421,'icir':0.267722,'hit_ratio':0.583333},'2027_onward_10d':{'ic_dates':1189,'ic':-0.005734,'icir':-0.017525,'hit_ratio':0.487805},'interpretation':'The specified 20-session primary horizon remains above both shared gates (absolute IC 0.0070; absolute ICIR 0.0840), and library overlap remains below 0.5000. It is retained as a low-weight medium-horizon diversifier. The post-2027 10-session diagnostic is negative and 1-/5-day decay is ineffective, so its next review is expedited and it should not be used as a short-horizon standalone signal.'}
d['last_validated']='2031-08-07T09:30:00';d['validation_timestamp']='2031-08-07T09:30:00'
d['benchmark_admission']['selected_metrics']={'ic':0.030611,'icir':0.093110,'metric_path':'validation.metrics (20d primary horizon)','max_abs_library_correlation':0.429992,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.002850174210}
with open(path,'w',encoding='utf8') as fh: json.dump(d,fh,indent=2);fh.write('\n')
print('updated',path,d['validation']['status'],d['last_validated'])

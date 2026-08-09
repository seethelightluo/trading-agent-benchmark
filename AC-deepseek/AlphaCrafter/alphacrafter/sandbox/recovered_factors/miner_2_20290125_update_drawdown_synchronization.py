"""Update persisted drawdown-synchronization factor after 2029-01-25 revalidation."""
import json
path='factors/miner_2_20261203_drawdown_synchronization_improvement_60_20.json'
d=json.load(open(path))
m=d['validation']['metrics']
m.update({'primary_horizon_days':20,'daily_paper_ic':0.041843,'daily_paper_icir':0.137382,'ic_std':0.304572,'ic_standard_error':0.012372,'ic_hit_ratio':0.590759,'ic_dates':606,'universe_instruments':15,'mean_valid_instruments_per_ic_date':13.069307,'signal_cell_coverage':0.282130,'mean_rank_turnover':0.087859,'turnover_dates':625,'max_abs_library_correlation':0.211786,'max_abs_library_correlation_factor':'miner_2_market_synchronization_increase_60_20','max_abs_library_correlation_common_cells':12899,'library_factors_screened':27,'decay':{'1d':{'ic':-0.013841,'icir':-0.043461,'hit_ratio':0.465600,'ic_dates':625,'mean_valid_instruments':13.067200},'5d':{'ic':-0.021364,'icir':-0.068679,'hit_ratio':0.481481,'ic_dates':621,'mean_valid_instruments':13.067633},'10d':{'ic':-0.019792,'icir':-0.064402,'hit_ratio':0.477273,'ic_dates':616,'mean_valid_instruments':13.068182},'20d':{'ic':0.041843,'icir':0.137382,'hit_ratio':0.590759,'ic_dates':606,'mean_valid_instruments':13.069307}}})
d['version']='2029-01-25'
d['validation'].update({'period':'2020-01-01 through 2029-01-24; visibility-safe completed-close cutoff','timestamp':'2029-01-25T09:30:00','status':'EFFECTIVE','regime_notes':{'2025_2026_20d':{'ic_dates':87,'ic':0.034432,'icir':0.131549,'hit_ratio':0.597701},'2027_to_2029_01_24_20d':{'ic_dates':519,'ic':0.043085,'icir':0.138367,'hit_ratio':0.589595},'interpretation':'The pre-specified 20-session horizon remains positive in the full history and both regime partitions, clearing the binding IC and ICIR gates. It has modestly softened from the prior revalidation but remains stable. Short horizons are negative/uninformative; use only as a medium-horizon diversifier.'}})
d['last_validated']='2029-01-25T09:30:00';d['validation_timestamp']='2029-01-25T09:30:00'
d['benchmark_admission']['selected_metrics']={'ic':0.043085,'icir':0.138367,'metric_path':'validation.regime_notes.2027_to_2029_01_24_20d','max_abs_library_correlation':0.211786,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.005960}
with open(path,'w') as f:json.dump(d,f,indent=2);f.write('\n')
print('updated',path,d['validation']['status'])

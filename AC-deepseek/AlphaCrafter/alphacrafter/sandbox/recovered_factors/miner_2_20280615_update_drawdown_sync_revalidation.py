import json
path='factors/miner_2_20261203_drawdown_synchronization_improvement_60_20.json'
with open(path,encoding='utf-8') as fh: d=json.load(fh)
v=d['validation']; v.update({
 'period':'2020-01-01 through 2028-06-14; eligible IC observations begin after signal warm-up',
 'timestamp':'2028-06-15T09:30:00', 'status':'EFFECTIVE',
 'metrics': {
  'primary_horizon_days':20,'daily_paper_ic':0.045307,'daily_paper_icir':0.150386,
  'ic_std':0.301269,'ic_standard_error':0.014266,'ic_hit_ratio':0.609865,'ic_dates':446,
  'universe_instruments':15,'mean_valid_instruments_per_ic_date':13.09417,
  'signal_cell_coverage':0.249746,'mean_rank_turnover':0.089535,'turnover_dates':465,
  'max_abs_library_correlation':0.223936,
  'max_abs_library_correlation_factor':'miner_1_breadth_recovery_capture_60d',
  'library_correlation_evidence':{'miner_1_breadth_recovery_capture_60d':{'rho':0.223936,'common_cells':9218},'miner_2_market_synchronization_increase_60_20':{'rho':0.214063,'common_cells':10819},'miner_1_market_beta_contraction_60_20':{'rho':-0.101885,'common_cells':10819},'miner_1_residualized_return_autocorrelation_20d':{'rho':-0.082656,'common_cells':5171},'miner_2_downside_beta_improvement_120_20':{'rho':-0.076317,'common_cells':7785}},
  'decay_ic':{'1d':-0.007875,'5d':-0.0047,'10d':-0.008414,'20d':0.045307},
  'decay_icir':{'1d':-0.024977,'5d':-0.015283,'10d':-0.02624,'20d':0.150386},
  'decay_ic_dates':{'1d':465,'5d':461,'10d':456,'20d':446}},
 'regime_notes':{'2025_2026_20d':{'ic_dates':87,'ic':0.034432,'icir':0.131549,'hit_ratio':0.597701},'2027_to_2028_06_14_20d':{'ic_dates':359,'ic':0.047942,'icir':0.154469,'hit_ratio':0.612813},'interpretation':'The 20-session horizon clears both gates in the full sample and both reported regimes. It remains a medium-horizon-only diversifier because 1-, 5-, and 10-session relations are weakly negative. Revalidate by 2028-09-15.'}
})
d['version']='2028-06-15';d['last_validated']='2028-06-15T09:30:00'
d['benchmark_admission']['selected_metrics']={'ic':0.045307,'icir':0.150386,'metric_path':'validation.metrics (20d)','max_abs_library_correlation':0.223936,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.006813535302}
with open(path,'w',encoding='utf-8') as fh: json.dump(d,fh,indent=2);fh.write('\n')
print('updated',path)

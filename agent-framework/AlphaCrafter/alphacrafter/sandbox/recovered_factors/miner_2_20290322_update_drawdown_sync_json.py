import json
path='factors/miner_2_20261203_drawdown_synchronization_improvement_60_20.json'
with open(path) as fh: d=json.load(fh)
d['version']='2029-03-22'
d['validation']['period']='2020-01-01 through 2029-03-21; visibility-safe completed-close cutoff'
d['validation']['timestamp']='2029-03-22T09:30:00'
d['validation']['status']='EFFECTIVE'
m=d['validation']['metrics']
m.update({'primary_horizon_days':20,'daily_paper_ic':0.062635,'daily_paper_icir':0.200589,'ic_std':0.312256,'ic_standard_error':0.012286,'ic_hit_ratio':0.611455,'ic_dates':646,'universe_instruments':15,'mean_valid_instruments_per_ic_date':13.065015,'signal_cell_coverage':0.289702,'mean_rank_turnover':0.086367,'turnover_dates':665,'max_abs_library_correlation':0.216391,'max_abs_library_correlation_factor':'miner_2_market_synchronization_increase_60_20','max_abs_library_correlation_common_cells':13419,'library_factors_screened':27,'decay':{'1d':{'ic':-0.011004,'icir':-0.034622,'hit_ratio':0.473684,'ic_dates':665,'mean_valid_instruments':13.063158},'5d':{'ic':-0.015238,'icir':-0.04896,'hit_ratio':0.488654,'ic_dates':661,'mean_valid_instruments':13.06354},'10d':{'ic':-0.009134,'icir':-0.029137,'hit_ratio':0.489329,'ic_dates':656,'mean_valid_instruments':13.064024},'20d':{'ic':0.062635,'icir':0.200589,'hit_ratio':0.611455,'ic_dates':646,'mean_valid_instruments':13.065015}}})
d['validation']['regime_notes']={'2025_2026_20d':{'ic_dates':87,'ic':0.034432,'icir':0.131549,'hit_ratio':0.597701},'2027_to_2029_03_21_20d':{'ic_dates':559,'ic':0.067024,'icir':0.209858,'hit_ratio':0.613596},'interpretation':'The pre-specified 20-session horizon strengthened and clears the binding IC and ICIR gates in both observed regime partitions. Short horizons remain negative/uninformative; use only as a medium-horizon diversifier.'}
d['last_validated']='2029-03-22T09:30:00';d['validation_timestamp']='2029-03-22T09:30:00'
d['benchmark_admission']['selected_metrics']={'ic':0.062635,'icir':0.200589,'metric_path':'validation.metrics','max_abs_library_correlation':0.216391,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.012563907115}
with open(path,'w') as fh: json.dump(d,fh,indent=2);fh.write('\n')
print('updated',path)

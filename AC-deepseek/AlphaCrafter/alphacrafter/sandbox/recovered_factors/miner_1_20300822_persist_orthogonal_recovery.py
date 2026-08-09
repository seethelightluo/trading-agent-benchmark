import json, os
record={
 'factor_id':'miner_1_20300822_orthogonal_recovery_acceleration_60_5_vs_ret5_ret20',
 'factor_name':'Orthogonal Recovery Acceleration', 'version':'2030-08-22',
 'calculation':{'expression':'cross_sectional_residual((recovery_position_60[t]-recovery_position_60[t-5]) ~ 1 + return_5 + return_20), lagged one completed day','description':'The change in each asset’s trailing 60-observation range position, with contemporaneous 5- and 20-observation returns removed by a daily cross-sectional regression. Positive values identify unusually improving recovery not explained by ordinary momentum.'},
 'dependencies':['daily close'], 'parameters':{'range_lookback':60,'acceleration_lag':5,'residual_controls':['return_5','return_20'],'min_periods':40,'minimum_cross_section':8,'validated_horizon_days':20},
 'validation':{'period':'2020-01-01 through 2030-08-21; point-in-time lagged observations','metrics':{
  'ic':0.040814,'icir':0.134154,'daily_paper_ic':0.040814,'daily_paper_icir':0.134154,'ic_hit_ratio':0.5527,'ic_dates':1916,'n_dates':3458,'mean_valid_instruments':11.88,'n_instruments':15,'minimum_instruments_per_ic_date':8,'coverage':0.457644,'turnover':0.181879,
  'decay':{'1d_ic':0.007037,'5d_ic':0.010187,'10d_ic':0.024611,'20d_ic':0.040814},
  'regime_notes':'20-day signal is positive in 2020-23 (IC 0.0125 at 5d; 0.0518 at 10d), mixed in 2024-27, and strongest in 2028 onward (5d IC 0.0336, ICIR 0.1115; 10d IC 0.0315, ICIR 0.1047). Latest 120 observations remain strong (10d IC 0.0922, ICIR 0.3491). Small cross-section implies conservative uncertainty interpretation.',
  'max_abs_library_correlation':0.443614,'max_library_correlation_factor':'raw_recovery_acceleration_control','library_correlation_evidence':{'all_admitted_factor_evidence_present':True,'scope':'pooled date-asset Spearman audit against 23 admitted factor signal reconstructions through 2030-08-21; maximum absolute correlation conservatively reported including the unreduced recovery-acceleration control','common_cells_max_pair':23738}
 },'status':'EFFECTIVE'},
 'tags':['recovery','cross-sectional-residual','orthogonal','trend','cross-asset'], 'last_validated':'2030-08-22',
 'benchmark_admission':{'contract':{'ic_threshold':0.007,'icir_threshold':0.084,'correlation_threshold':0.5,'library_capacity':30,'active_top_k':10},'selected_metrics':{'ic':0.040814,'icir':0.134154,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.443614,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.005476},'admitted_at':'2030-08-22T00:00:00Z'}
}
path='factors/miner_1_20300822_orthogonal_recovery_acceleration_60_5_vs_ret5_ret20.json'
with open(path,'w') as f: json.dump(record,f,indent=2)
print(json.dumps({'persisted':path,'status':'EFFECTIVE','max_abs_library_correlation':0.443614}))

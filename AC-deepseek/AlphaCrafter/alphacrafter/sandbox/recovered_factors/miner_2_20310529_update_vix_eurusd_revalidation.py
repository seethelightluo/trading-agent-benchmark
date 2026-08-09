import json
path='factors/miner_2_20310220_inverse_residual_vix_eurusd_joint_stress_loading_expansion_20_60d.json'
with open(path,encoding='utf8') as fh: d=json.load(fh)
d['validation']={
 'period':'2020-01-01 through 2031-05-28; completed-close visibility-safe cutoff',
 'status':'EFFECTIVE','selected_horizon_days':20,
 'metrics':{'daily_paper_ic':0.063674,'daily_paper_icir':0.209013,'ic_std':0.304643,'ic_standard_error':0.008813,'ic_hit_ratio':0.604184,'ic_dates':1195,'mean_valid_instruments':14.991632,'universe_instruments':15,'coverage':0.334099,'mean_daily_rank_turnover':0.126032,'turnover_dates':1214,'joint_stress_driver_nonzero_fraction':0.127939,'max_abs_library_correlation':0.308222,'most_correlated_library_factor':'miner_2_dxy_shock_beta_improvement_60_20','library_correlation_cells':18310,'library_factors_screened':27,'decay':{'1d':{'ic':0.01391,'icir':0.042659,'ic_dates':1214},'5d':{'ic':0.032458,'icir':0.100108,'ic_dates':1210},'10d':{'ic':0.057065,'icir':0.181457,'ic_dates':1205},'20d':{'ic':0.063674,'icir':0.209013,'ic_dates':1195}}},
 'regime_notes':'At 10 days, 2025-2026: 66 IC dates, IC 0.080985, ICIR 0.334300, hit ratio 0.696970; 2027 onward: 1,139 IC dates, IC 0.055679, ICIR 0.174984, hit ratio 0.584723. Exact joint-stress driver is nonzero on 12.7939% of daily observations. Both eligible regimes remain positive; full-sample 20-day horizon passes IC and ICIR gates.',
 'timestamp':'2031-05-29T09:30:00'}
d['last_validated']='2031-05-29';d['validation_timestamp']='2031-05-29'
d['benchmark_admission']['selected_metrics']={'ic':0.063674,'icir':0.209013,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.308222,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.013308799662}
with open(path,'w',encoding='utf8') as fh: json.dump(d,fh,ensure_ascii=False,indent=2);fh.write('\n')
print('UPDATED',path,'status',d['validation']['status'],'last_validated',d['last_validated'])

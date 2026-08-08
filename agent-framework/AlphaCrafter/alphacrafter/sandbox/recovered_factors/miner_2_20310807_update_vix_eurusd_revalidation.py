import json
path='factors/miner_2_20310220_inverse_residual_vix_eurusd_joint_stress_loading_expansion_20_60d.json'
with open(path,encoding='utf8') as fh: x=json.load(fh)
m=x['validation']['metrics']
m.update({'daily_paper_ic':0.05636,'daily_paper_icir':0.183085,'ic_std':0.307837,'ic_standard_error':0.008724,'ic_hit_ratio':0.595984,'ic_dates':1245,'mean_valid_instruments':14.991968,'universe_instruments':15,'coverage':0.343078,'valid_factor_cells':19082,'mean_daily_rank_turnover':0.126508,'turnover_dates':1264,'joint_stress_driver_nonzero_fraction':0.12972,'max_abs_library_correlation':0.305138,'most_correlated_library_factor':'miner_2_dxy_shock_beta_improvement_60_20','library_correlation_cells':19060,'library_factors_screened':27,'decay':{'1d':{'ic':0.012715,'icir':0.039099,'ic_dates':1264},'5d':{'ic':0.031147,'icir':0.096548,'ic_dates':1260},'10d':{'ic':0.052921,'icir':0.167734,'ic_dates':1255},'20d':{'ic':0.05636,'icir':0.183085,'ic_dates':1245}}})
x['validation']['period']='2020-01-01 through 2031-08-06; completed-close visibility-safe cutoff'
x['validation']['status']='EFFECTIVE'; x['validation']['selected_horizon_days']=20
x['validation']['regime_notes']='At 10 days, 2025-2026: 66 IC dates, IC 0.080985, ICIR 0.334300, hit ratio 0.696970; 2027 onward: 1,189 IC dates, IC 0.051363, ICIR 0.160969, hit ratio 0.582002. Joint VIX-up/EURUSD-down driver is nonzero on 12.9720% of observations. The selected 20-day horizon remains above both admission gates; 2027-onward performance remains positive, though softer than the earlier validation.'
x['validation']['timestamp']='2031-08-07T09:30:00'; x['last_validated']='2031-08-07'; x['validation_timestamp']='2031-08-07'
b=x['benchmark_admission']['selected_metrics']; b.update({'ic':0.05636,'icir':0.183085,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.305138,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.0103196706})
with open(path,'w',encoding='utf8') as fh: json.dump(x,fh,ensure_ascii=False,indent=2);fh.write('\n')
print('updated',path,'status',x['validation']['status'],'validated',x['last_validated'])

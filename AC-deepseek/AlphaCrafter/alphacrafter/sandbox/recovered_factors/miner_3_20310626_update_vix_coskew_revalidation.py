import json
p='factors/miner_3_20290405_vix_stress_residual_downside_coskewness_contraction_20_60d.json'
d=json.load(open(p,encoding='utf8'))
d['version']='2031-06-26'
d['last_validated']='2031-06-26'
d['validation_timestamp']='2031-06-26T00:00:00'
d['validation']['period']='2020-01-01 through 2031-06-25; completed-close visibility-safe cutoff'
d['validation']['status']='EFFECTIVE'
d['validation']['metrics']={
 'selected_horizon_days':10,'daily_paper_ic':0.033279,'daily_paper_icir':0.098611,'ic_std':0.337479,'ic_standard_error':0.009754,'ic_hit_ratio':0.513784,'ic_dates':1197,'mean_valid_instruments':14.991646,'coverage':0.421316,'mean_daily_rank_turnover':0.054558,'turnover_dates':1234,'stress_day_fraction':0.471724,'max_abs_library_correlation':0.281714,'most_correlated_library_factor':'miner_3_residual_downside_volume_confirmation_deceleration_20_60d','library_factors_screened':27,'library_correlation_cells':15833,
 'decay':{'1d':{'ic':0.021842,'icir':0.068035,'ic_dates':1206,'hit_ratio':0.510779,'mean_valid_instruments':14.991708},'5d':{'ic':0.025396,'icir':0.078339,'ic_dates':1202,'hit_ratio':0.513311,'mean_valid_instruments':14.991681},'10d':{'ic':0.033279,'icir':0.098611,'ic_dates':1197,'hit_ratio':0.513784,'mean_valid_instruments':14.991646},'20d':{'ic':0.019799,'icir':0.056415,'ic_dates':1187,'hit_ratio':0.511373,'mean_valid_instruments':14.991575}}
}
d['validation']['regime_notes']='Decision-aligned 10-day horizon remains admissible: IC +0.033279 and ICIR +0.098611. Maximum absolute pooled asset-date Spearman correlation is 0.281714 against 27 reconstructed admitted-library signals (15,833 paired cells), below 0.500000. The 2025-2026 subset is adverse (66 dates: IC -0.025504, ICIR -0.091568, hit 30.30%), while 2027 onward is positive (1,131 dates: IC +0.036710, ICIR +0.107844, hit 52.61%). Conditional coverage is 42.13%; retain as a stress-diversification sleeve and revalidate quarterly.'
d['benchmark_admission']['selected_metrics']={'ic':0.033279,'icir':0.098611,'metric_path':'validation.metrics.decay.10d','max_abs_library_correlation':0.281714,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.003281673669}
open(p,'w',encoding='utf8').write(json.dumps(d,indent=2,ensure_ascii=False)+'\n')

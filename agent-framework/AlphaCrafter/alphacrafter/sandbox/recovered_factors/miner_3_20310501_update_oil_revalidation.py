import json
path='factors/miner_3_20300822_residual_positive_oil_change_shock_loading_contraction_20_60d.json'
with open(path,encoding='utf8') as fh: x=json.load(fh)
m={'daily_paper_ic':0.057749,'daily_paper_icir':0.181704,'ic_std':0.317819,'ic_standard_error':0.009233,'ic_hit_ratio':0.544304,'ic_dates':1185,'mean_valid_instruments':14.991561,'coverage':0.329852,'mean_rank_turnover':0.108964,'turnover_dates':1194,'max_abs_library_correlation':0.209718,'most_correlated_library_factor':'miner_1_market_beta_contraction_60_20','library_correlation_cells':18000,'decay':{'1d':{'ic':-0.007321,'icir':-0.022961,'dates':1194},'5d':{'ic':0.041561,'icir':0.128485,'dates':1190},'10d':{'ic':0.057749,'icir':0.181704,'dates':1185},'20d':{'ic':0.052057,'icir':0.158259,'dates':1175}}}
x['validation']['period']='2020-01-01 to 2031-04-30; valid IC observations begin after required factor history'
x['validation']['status']='EFFECTIVE'; x['validation']['metrics']=m
x['validation']['regime_notes']='Revalidation at the 2031-04-30 data cutoff: decision-aligned 10-day IC is +0.057749 (ICIR +0.181704; 1,185 dates; 14.991561 mean valid instruments); it clears both binding gates. 2025-2026 (66 dates) remained adverse (IC -0.089912, ICIR -0.344421, hit 0.318182), while 2027 onward (1,119 dates) was robust (IC +0.066458, ICIR +0.208450, hit 0.557641). Signal coverage is 32.9852%, rank turnover 0.108964, and maximum library Spearman overlap is 0.209718, below the 0.5000 ceiling.'
x['validation']['validation_timestamp']='2031-05-01';x['last_validated']='2031-05-01';x['validation_timestamp']='2031-05-01'
x['benchmark_admission']['selected_metrics'].update({'ic':0.057749,'icir':0.181704,'max_abs_library_correlation':0.209718,'quality':0.057749*0.181704})
with open(path,'w',encoding='utf8') as fh: json.dump(x,fh,indent=2);fh.write('\n')

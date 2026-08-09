import json
from pathlib import Path
p=Path('factors/miner_1_20280127_residualized_inflation_basket_correlation_decoupling_60_20.json')
d=json.loads(p.read_text())
m=d['validation']['metrics']
m.update({'daily_paper_ic':0.042978,'daily_paper_icir':0.140543,'ic_std':0.305797,'ic_standard_error':0.013359,'ic_hit_ratio':0.559160,'ic_dates':524,'mean_valid_instruments':11.091603,'coverage':0.134377,'rank_turnover':0.122800,'turnover_dates':532,'concentration':'continuous residual score; mean 11.09 valid names on eligible dates','max_abs_library_correlation':0.313649,'most_correlated_library_factor':'miner_1_market_beta_contraction_60_20','library_correlation_cells':5922,'library_factors_compared':24,'decay':{'1d':{'ic':-0.004104,'icir':-0.014028,'ic_dates':533,'hit_ratio':0.484053},'5d':{'ic':0.044898,'icir':0.152396,'ic_dates':529,'hit_ratio':0.561437},'10d':{'ic':0.042978,'icir':0.140543,'ic_dates':524,'hit_ratio':0.559160},'20d':{'ic':0.027326,'icir':0.089643,'ic_dates':514,'hit_ratio':0.531128}}})
d['version']='2028-08-24'
d['validation']['period']='2020-01-01 through 2028-08-23; point-in-time cutoff 2028-08-23'
d['validation']['regime_notes']='All 524 eligible 10-session IC observations are in the available 2026-28 synthetic segment after rolling-history and coverage constraints; no independently eligible pre-2026 observations exist. The factor is therefore regime-limited despite a larger sample. Five-, 10-, and 20-session horizons pass benchmark gates; 5-session is strongest on this cutoff.'
d['validation']['admission_basis']='Revalidation: 10-session |IC|=0.042978 >= 0.007000; |ICIR|=0.140543 >= 0.084000; maximum absolute Spearman library correlation=0.313649 < 0.500000. (5d and 20d also pass.)'
d['validation']['revalidation_history'].append({'date':'2028-08-24','cutoff':'2028-08-23','status':'EFFECTIVE','selected_horizon_sessions':10,'ic':0.042978,'icir':0.140543,'max_abs_library_correlation':0.313649,'library_factors_compared':24})
d['last_validated']='2028-08-24'
b=d['benchmark_admission']['selected_metrics'];b.update({'ic':0.042978,'icir':0.140543,'max_abs_library_correlation':0.313649,'quality':0.006040254054})
p.write_text(json.dumps(d,indent=2)+'\n')
print('updated',p,'status',d['validation']['status'],'quality',b['quality'])

import json
from pathlib import Path
p=Path('factors/miner_2_20280601_residual_upside_serial_reversal_60d.json')
d=json.loads(p.read_text())
d['version']='2028-09-07'
d['validation']['period']='2020-01-01 through 2028-09-06; eligible IC observations begin in 2025 because of aligned-calendar coverage and rolling requirements'
d['validation']['timestamp']='2028-09-07T09:30:00'
d['validation']['status']='EFFECTIVE'
m=d['validation']['metrics']
m.update({'primary_horizon_days':20,'daily_paper_ic':-0.058435,'daily_paper_icir':-0.191829,'ic_std':0.304620,'ic_standard_error':0.013889,'ic_hit_ratio':0.424116,'ic_dates':481,'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.5842,'signal_cell_coverage':0.226594,'mean_rank_turnover':0.041513,'turnover_dates':500,'concentration_note':'Continuous residual-autocorrelation signal; 481 IC dates meet the at-least-8-name rule, averaging 10.58 valid names; latest signal has 10 valid instruments.','max_abs_library_correlation':0.193710,'max_abs_library_correlation_factor':'miner_3_residual_lower_partial_moment_60d','library_signal_cells_compared':10020,'decay':{'1d':{'ic':-0.006950,'icir':-0.020791,'hit_ratio':0.488000,'dates':500,'mean_valid_instruments':10.5620},'5d':{'ic':-0.028901,'icir':-0.085948,'hit_ratio':0.459677,'dates':496,'mean_valid_instruments':10.5665},'10d':{'ic':-0.037298,'icir':-0.114548,'hit_ratio':0.446029,'dates':491,'mean_valid_instruments':10.5723},'20d':{'ic':-0.058435,'icir':-0.191829,'hit_ratio':0.424116,'dates':481,'mean_valid_instruments':10.5842}}})
d['validation']['regime_notes']={'2025_2026':{'ic_20d':-0.113459,'icir_20d':-0.435818,'hit_ratio_20d':0.370968,'ic_dates':62},'2027_2028_09_06':{'ic_20d':-0.050293,'icir_20d':-0.162193,'hit_ratio_20d':0.431981,'ic_dates':419},'interpretation':'Negative 20-session orientation remains effective in both eligible periods. The recent-regime magnitude has moderated, but 20-day absolute IC and ICIR remain above admission gates and maximum library overlap remains low. Retain as a diversifying medium-horizon signal and revalidate within three months.'}
d['last_validated']='2028-09-07T09:30:00'
d['benchmark_admission']['selected_metrics']={'ic':-0.058435,'icir':-0.191829,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.193710,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.011210164365}
p.write_text(json.dumps(d,indent=2)+'\n')
print('UPDATED',p,'STATUS',d['validation']['status'],'IC',m['daily_paper_ic'],'ICIR',m['daily_paper_icir'],'CORR',m['max_abs_library_correlation'])

"""Update persisted USDJPY-transition factor validation using current visible cutoff."""
import json
p='factors/miner_1_20300502_usdjpy_volatility_transition_beta_resilience_60.json'
with open(p,encoding='utf-8') as f: x=json.load(f)
x['validation']['period']='2020-01-01 through 2030-05-29 visible-data cutoff'
x['validation']['status']='EFFECTIVE'
m=x['validation']['metrics']
m.update({'coverage_cells':5505,'total_cells':50970,'coverage':0.108005,'mean_rank_change_turnover':0.038552,'mean_cross_sectional_std':0.625506,
'forward_1_session':{'ic_dates':366,'ic':0.013125,'icir':0.042017,'hit_ratio':0.478142,'mean_instruments':15.0,'minimum_instruments':15},
'forward_5_session':{'ic_dates':364,'ic':0.025801,'icir':0.080976,'hit_ratio':0.530220,'mean_instruments':15.0,'minimum_instruments':15},
'forward_10_session':{'ic_dates':364,'ic':0.056258,'icir':0.180231,'hit_ratio':0.541209,'mean_instruments':15.0,'minimum_instruments':15},
'forward_20_session':{'ic_dates':363,'ic':0.085135,'icir':0.295882,'hit_ratio':0.586777,'mean_instruments':15.0,'minimum_instruments':15},
'recent_180_calendar_days_10_session':{'ic_dates':44,'ic':-0.083250,'icir':-0.368096,'hit_ratio':0.295455},
'max_abs_library_correlation':0.216632,'most_correlated_library_factor':'delayed_post_stress_relative_rebound_reversal_60','library_correlation_paired_cells':3195})
x['validation']['regime_notes']='Full-history validation remains above the binding gates at 10 and 20 sessions; strongest selected 20-session result is IC +0.085135 and ICIR +0.295882. The 5-session ICIR (+0.080976) is below the gate. 2025-2026 evidence remains adverse (20 eligible 10-session dates, IC -0.056351, ICIR -0.250705); 2027 through 2030-05-29 is positive (344 dates, IC +0.062805, ICIR +0.199071). The recent 180-calendar-day 10-session window is materially adverse (44 dates, IC -0.083250, ICIR -0.368096). The factor remains admitted based on full-history validation but should be used cautiously and revalidated before the next quarterly review. Exact audit against all 30 admitted-library reconstructions found max absolute pooled Spearman correlation +0.216632 (3,195 paired cells), below 0.5000.'
x['last_validated']='2030-05-30'
x['benchmark_admission']['selected_metrics'].update({'ic':0.085135,'icir':0.295882,'metric_path':'validation.metrics.forward_20_session','max_abs_library_correlation':0.216632,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.025189})
with open(p,'w',encoding='utf-8') as f: json.dump(x,f,indent=2);f.write('\n')
print('UPDATED',p,'status',x['validation']['status'],'last_validated',x['last_validated'])

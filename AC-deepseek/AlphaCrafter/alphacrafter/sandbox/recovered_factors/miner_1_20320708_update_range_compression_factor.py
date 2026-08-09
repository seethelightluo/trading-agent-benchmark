import json
p='factors/miner_1_20310626_residual_downside_range_compression_persistence_20_60obs.json'
with open(p,encoding='utf-8') as f: d=json.load(f)
old=d['validation']
new_metrics={
 'selected_horizon_days':20,'daily_paper_ic':0.059396,'daily_paper_icir':0.177059,
 'hit_ratio':0.57839,'ic_dates':1416,'mean_valid_instruments_per_ic_date':9.501,
 'minimum_valid_instruments_per_ic_date':8,'coverage':0.585256,
 'valid_signal_cells':13695,'total_signal_cells':23400,'mean_rank_turnover':0.080172,
 'turnover_comparisons':1559,'concentration_median_iqr':0.084309,
 'max_abs_library_correlation':0.459838,'closest_library_factor':'relative_liquidity_stress_20_60obs',
 'library_correlation_paired_cells':12950,'library_signals_screened':30,
 'library_correlation_note':'Unchanged-factor revalidation. The complete admission-time library Spearman audit remains evidence; no novel admission is requested.',
 'decay':{
  '1d':{'ic':-0.002986,'icir':-0.00867,'hit_ratio':0.504219,'ic_dates':1422},
  '5d':{'ic':0.012546,'icir':0.036266,'hit_ratio':0.511283,'ic_dates':1418},
  '10d':{'ic':0.050957,'icir':0.148998,'hit_ratio':0.558616,'ic_dates':1416},
  '20d':{'ic':0.059396,'icir':0.177059,'hit_ratio':0.57839,'ic_dates':1416}
 }
}
# preserve prior validation before replacement
hist={'period':old['period'],'status':old['status'],'metrics':old['metrics'],'regime_notes':old['regime_notes']}
d.setdefault('validation_history',[]).append(hist)
d['validation']={'period':'2026-07-16 through 2032-07-07 completed daily bars; forward-return availability varies by horizon','status':'EFFECTIVE','metrics':new_metrics,'regime_notes':'At the selected 20-day horizon, aggregate evidence remains above shared gates and is positive in both broad partitions: 2026-2029 (831 IC dates, IC 0.054259, ICIR 0.163803, hit 57.160%) and 2030-2032-07-07 (585 dates, IC 0.066692, ICIR 0.195291, hit 58.803%). Recent-12-month evidence has softened (194 dates, IC 0.026890, ICIR 0.075877, hit 53.608%) and does not independently pass the ICIR gate. Retain as EFFECTIVE under enhanced monitoring because the full out-of-sample aggregate passes at 10 and 20 days; calculation and prior complete library novelty evidence are unchanged.'}
d['last_validated']='2032-07-08'; d['next_revalidation_due']='2032-10-08'
d['benchmark_admission']['selected_metrics'].update({'ic':0.059396,'icir':0.177059,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.459838,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.010516})
with open(p,'w',encoding='utf-8') as f: json.dump(d,f,indent=2); f.write('\n')
print('updated',p,'history_entries',len(d['validation_history']))

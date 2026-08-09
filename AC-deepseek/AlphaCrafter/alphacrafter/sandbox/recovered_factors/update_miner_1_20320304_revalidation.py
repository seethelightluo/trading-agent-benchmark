import json
p='factors/miner_1_20310626_residual_downside_range_compression_persistence_20_60obs.json'
with open(p) as h: d=json.load(h)
old={k:d['validation'][k] for k in ('period','status','metrics','regime_notes')}
d['validation_history'].append(old)
m={'selected_horizon_days':20,'daily_paper_ic':0.07285,'daily_paper_icir':0.221321,'hit_ratio':0.593391,'ic_dates':1392,'mean_valid_instruments_per_ic_date':9.55,'minimum_valid_instruments_per_ic_date':8,'coverage':0.315051,'valid_signal_cells':18232,'total_signal_cells':57870,'mean_rank_turnover':0.104776,'turnover_comparisons':1410,'concentration_median_iqr':0.07034,'max_abs_library_correlation':0.459838,'closest_library_factor':'relative_liquidity_stress_20_60obs','library_correlation_paired_cells':12950,'library_signals_screened':30,'library_correlation_note':'Calculation is unchanged. Admission-time complete-library novelty audit remains valid evidence; no novel candidate admission is being requested.','decay':{'1d':{'ic':-0.002301,'icir':-0.006666,'hit_ratio':0.510985,'ic_dates':1411},'5d':{'ic':0.014564,'icir':0.042254,'hit_ratio':0.512438,'ic_dates':1407},'10d':{'ic':0.057749,'icir':0.169591,'hit_ratio':0.564907,'ic_dates':1402},'20d':{'ic':0.07285,'icir':0.221321,'hit_ratio':0.593391,'ic_dates':1392}}}
d['validation']={'period':'2026-07-16 through 2032-03-03 completed daily bars; forward-return availability varies by horizon','status':'EFFECTIVE','metrics':m,'regime_notes':'The selected 20-day signal remained positive and improved in both partitions: 2026-2029 (849 IC dates, IC 0.059243, ICIR 0.179506, hit 57.833%) and 2030-2032-03-03 (543 dates, IC 0.094126, ICIR 0.287878, hit 61.694%). It passes shared IC/ICIR gates at 10 and 20 days. Definition unchanged; original complete-library Spearman novelty evidence remains 0.459838, below 0.5000.'}
d['last_validated']='2032-03-04'; d['next_revalidation_due']='2032-06-04'
d['benchmark_admission']['selected_metrics'].update({'ic':0.07285,'icir':0.221321,'quality':0.01612377485})
with open(p,'w') as h: json.dump(d,h,indent=2); h.write('\n')
print('updated',p,'history',len(d['validation_history']))

import json
p='factors/miner_1_20310403_inverse_residual_downside_range_expansion_exhaustion_60obs.json'
d=json.load(open(p)); v=d['validation']; m=v['metrics']
v['period']='2020-01-01 through 2031-05-28 (visible-through date)'
m.update({'selected_horizon_days':20,'daily_paper_ic':0.071670,'daily_paper_icir':0.216477,'hit_ratio':0.59599,'ic_dates':1198,'mean_valid_instruments':9.74,'signal_cell_coverage':0.302770,'signal_cells':'16613/54870','daily_rank_turnover':0.043155,'concentration_mean_cross_sectional_sd':0.067372,'decay':{'1d':{'ic':0.011661,'icir':0.033721,'dates':1217},'5d':{'ic':0.018452,'icir':0.052221,'dates':1213},'10d':{'ic':0.027522,'icir':0.077900,'dates':1208},'20d':{'ic':0.071670,'icir':0.216477,'dates':1198}}})
r=v['regime_notes']; r['2026_current']={'dates':1198,'ic':0.071670,'icir':0.216477,'hit_ratio':0.59599}; r['assessment']='Revalidation retains admission at the 20-day horizon: both shared IC gates remain satisfied and have improved versus 2031-05-15. All usable observations remain post-2026, so evidence is statistically substantial but regime breadth remains limited. The original complete 30-signal admission novelty audit (maximum correlation 0.348663) remains the recorded library evidence; no new admission was made. Retain conservatively and revalidate by 2031-08-29.'
d['last_validated']='2031-05-29'
open(p,'w').write(json.dumps(d,indent=2)+'\n')
print('updated',p)

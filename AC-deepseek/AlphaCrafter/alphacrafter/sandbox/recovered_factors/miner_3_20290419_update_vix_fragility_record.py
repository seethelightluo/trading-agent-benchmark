import json
p='factors/miner_3_20290405_inverse_residual_sustained_vix_relief_fragility_20_80obs.json'
with open(p) as h: d=json.load(h)
m=d['validation']['metrics']
m.update({'daily_paper_ic':0.0446179279,'daily_paper_icir':0.1537032849,'same_horizon_days':20,'hit_ratio':0.5659340659,'ic_dates':364,'mean_valid_instruments_per_ic_date':11.989010989,'coverage_cells':4568,'coverage_total_cells':46620,'coverage':0.097983698,'turnover_10d':0.8260177323,'max_abs_library_correlation':0.0832880298,'max_library_correlation_factor':'orthogonal_trend_acceleration_20_60obs','library_correlation_evidence':'Pooled date-asset Spearman screen against 16 reconstructed admitted comparators through 2029-04-18. Maximum absolute correlation 0.0832880298 with orthogonal trend acceleration, based on 4,504 common cells.'})
m['decay']={'1d':{'ic':0.0092385684,'icir':0.0309884616,'dates':380},'5d':{'ic':0.0220223039,'icir':0.0759678632,'dates':376},'10d':{'ic':0.0171490684,'icir':0.0639659646,'dates':371},'20d':{'ic':0.0446179279,'icir':0.1537032849,'dates':364}}
d['validation']['period']='2020-01-01 through 2029-04-18, point-in-time research data'
d['validation']['regime_notes']='Selected 20-day horizon: 2026 IC/ICIR +0.07539/+0.32842 (71 dates), 2027 +0.02310/+0.06821 (56), 2028 +0.05107/+0.16822 (215), 2029 -0.06294/-0.40299 (22). Latest 120 mature dates: +0.11177/+0.39937. The full sample continues to exceed both admission gates, and the recent 120-date segment remains strong, but the adverse small 2029 segment warrants a modest sleeve and close revalidation.'
d['validation']['validation_timestamp']='2029-04-19T00:00:00';d['last_validated']='2029-04-19';d['version']='2029-04-19'
s=d['benchmark_admission']['selected_metrics'];s.update({'ic':0.0446179279,'icir':0.1537032849,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.0832880298,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.006858'} )
with open(p,'w') as h: json.dump(d,h,indent=2);h.write('\n')

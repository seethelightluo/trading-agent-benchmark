import json
from pathlib import Path
p=Path('factors/miner_1_20310403_inverse_residual_downside_range_expansion_exhaustion_60obs.json')
d=json.loads(p.read_text())
old=d['validation'].copy()
metrics={
 'selected_horizon_days':20,'daily_paper_ic':0.050087,'daily_paper_icir':0.141789,'hit_ratio':0.581167,'ic_dates':1423,
 'mean_valid_instruments':9.826,'minimum_valid_instruments':9,'signal_cell_coverage':0.624818,'signal_cells':'14199/22725',
 'daily_rank_turnover':0.043797,'concentration_median_cross_sectional_iqr':0.064704,
 'decay':{'1d':{'ic':0.004749,'icir':0.013727,'dates':1442},'5d':{'ic':0.010629,'icir':0.029801,'dates':1438},'10d':{'ic':0.019575,'icir':0.053582,'dates':1433},'20d':{'ic':0.050087,'icir':0.141789,'dates':1423}},
 'max_abs_library_correlation':0.348663,'closest_library_factor':'idiosyncratic_upside_tail_skewness_60obs','library_correlation_evidence_cells':8669,'library_factors_screened':30,
 'correlation_evidence_note':'Unchanged-factor revalidation; complete admission novelty audit is retained from prior validation.'}
d.setdefault('validation_history',[]).append(old)
d['version']='2031-11-27'
d['validation']={'period':'2026-01-01 through 2031-11-26 (completed-bar cutoff)','status':'EFFECTIVE','metrics':metrics,'regime_notes':{
 '2026_2029':{'dates':946,'ic':0.054705,'icir':0.169594,'hit_ratio':0.593023},
 '2030_2031_11_26':{'dates':477,'ic':0.040928,'icir':0.100427,'hit_ratio':0.557652},
 'recent_12m':{'dates':241,'ic':0.000346,'icir':0.000751,'hit_ratio':0.547718},
 'assessment':'The full post-2026 20-day validation passes both shared IC and ICIR gates and both broad partitions remain positive with positive ICIR. The recent 12-month contribution is effectively flat, so the factor remains EFFECTIVE under the aggregate revalidation rule but is on heightened drift monitoring and should be reassessed at the next quarterly cadence.'}}
d['last_validated']='2031-11-27'
d['next_revalidation_due']='2032-02-27'
p.write_text(json.dumps(d,indent=2)+'\n')
print('updated',p)

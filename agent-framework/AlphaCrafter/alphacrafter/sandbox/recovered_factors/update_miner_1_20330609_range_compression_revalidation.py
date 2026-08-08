"""Persist 2033-06-09 enhanced revalidation for admitted miner_1 factor."""
import json
p='factors/miner_1_20310626_residual_downside_range_compression_persistence_20_60obs.json'
d=json.load(open(p))
d['version']='2033-06-09'
d['last_validated']='2033-06-09'
d['next_revalidation_due']='2033-07-09'
metrics={
 'selected_horizon_days':20,'daily_paper_ic':0.049813,'daily_paper_icir':0.145098,'hit_ratio':0.563901,'ic_dates':1651,
 'mean_valid_instruments':9.309509,'minimum_valid_instruments':8,'universe_instruments':15,
 'signal_cell_coverage':0.326393,'signal_cells':'20504/62820','daily_rank_turnover':0.107678,
 'concentration_median_cross_sectional_iqr':0.070236,'constant_cross_sections':29,
 'max_abs_library_correlation':0.459838,'closest_library_factor':'relative_liquidity_stress_20_60obs','library_factors_screened':30,
 'correlation_evidence_note':'Unchanged-factor revalidation. Complete admission-time library Spearman audit remains applicable: maximum absolute rho 0.459838, below 0.500000.',
 'decay':{'1d':{'ic':-0.006021,'icir':-0.017192,'hit_ratio':0.503593,'dates':1670},'5d':{'ic':0.003982,'icir':0.011402,'hit_ratio':0.504802,'dates':1666},'10d':{'ic':0.038323,'icir':0.107576,'hit_ratio':0.543648,'dates':1661},'20d':{'ic':0.049813,'icir':0.145098,'hit_ratio':0.563901,'dates':1651}}
}
notes='Aggregate selected-20d evidence passes shared gates (IC 0.049813; ICIR 0.145098). Partitions: 2026-2029: 849 dates, IC 0.059243 / ICIR 0.179506; 2030-2033-05-11: 802 dates, IC 0.039830 / ICIR 0.111647. Latest 12 months: 217 dates, IC -0.032587 / ICIR -0.084303 / hit 44.24%. Retain EFFECTIVE based on complete history, but recent deterioration warrants enhanced monthly monitoring and no influence increase.'
d['validation']={'period':'2026-07-16 through 2033-05-11 completed daily bars for selected 20-day forward horizon; visible source data through 2033-06-08','status':'EFFECTIVE','metrics':metrics,'regime_notes':notes}
d['validation_history'].append({'date':'2033-06-09','cutoff':'2033-05-11','status':'EFFECTIVE','period':d['validation']['period'],'metrics':metrics,'regime_notes':notes,'note':'Enhanced monthly revalidation. Definition unchanged, complete forward windows only. Aggregate gates pass, while latest-12-month performance has deteriorated materially; retain only with enhanced monitoring.'})
with open(p,'w') as f: json.dump(d,f,indent=2); f.write('\n')
print('updated',p,'history',len(d['validation_history']))

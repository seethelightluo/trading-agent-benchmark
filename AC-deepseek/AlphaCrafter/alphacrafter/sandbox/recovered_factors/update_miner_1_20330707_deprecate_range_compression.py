import json
from pathlib import Path
p=Path('factors/miner_1_20310626_residual_downside_range_compression_persistence_20_60obs.json')
d=json.loads(p.read_text())
m={
 'selected_horizon_days':20,'daily_paper_ic':0.049202,'daily_paper_icir':0.143348,'hit_ratio':0.562537,'ic_dates':1671,
 'mean_valid_instruments':9.293836,'minimum_valid_instruments':8,'universe_instruments':15,
 'signal_cell_coverage':0.327361,'signal_cells':'20663/63120','daily_rank_turnover':0.107509,
 'concentration_median_cross_sectional_iqr':0.070413,'constant_cross_sections':29,
 'max_abs_library_correlation':0.459838,'closest_library_factor':'relative_liquidity_stress_20_60obs','library_factors_screened':30,
 'correlation_evidence_note':'Unchanged-factor revalidation. Complete admission-time library Spearman audit remains applicable: maximum absolute rho 0.459838, below 0.500000.',
 'decay':{'1d':{'ic':-0.00595,'icir':-0.016954,'hit_ratio':0.50355,'dates':1690},'5d':{'ic':0.005601,'icir':0.015978,'hit_ratio':0.505931,'dates':1686},'10d':{'ic':0.040742,'icir':0.114005,'hit_ratio':0.544914,'dates':1681},'20d':{'ic':0.049202,'icir':0.143348,'hit_ratio':0.562537,'dates':1671}}
}
d['version']='2033-07-07'; d['last_validated']='2033-07-07'; d['next_revalidation_due']=None
d['validation']={'period':'2026-07-16 through 2033-06-08 completed daily bars for selected 20-day forward horizon; visible source data through 2033-07-06','status':'DEPRECATED','metrics':m,'regime_notes':'Aggregate selected-20d evidence remains above the shared gates (IC 0.049202; ICIR 0.143348), and original unchanged-definition library novelty evidence remains 0.459838 < 0.500000. Nevertheless recent predictive performance remains unambiguously failed: latest 12 months (237 dates) IC -0.029938 / ICIR -0.078277 / hit 44.30%; latest 6 months (122 dates) IC -0.134856 / ICIR -0.367911 / hit 32.79%. Under enhanced-drift policy, the repeated materially negative trailing ICIR warrants deprecation rather than allowing broad historical evidence to sustain active use.'}
d['validation_history'].append({'date':'2033-07-07','cutoff':'2033-06-08','status':'DEPRECATED','period':d['validation']['period'],'metrics':m,'regime_notes':d['validation']['regime_notes'],'note':'Enhanced revalidation with completed forward windows only. Although aggregate admission gates still pass, the second consecutive severe negative six-month regime and negative twelve-month regime fail timeliness requirements; remove from active library pending a future fresh re-admission.'})
out=Path('factors/miner_1_20310626_residual_downside_range_compression_persistence_20_60obs_deprecated.json')
out.write_text(json.dumps(d,indent=2)+'\n')
p.unlink()
print(out)

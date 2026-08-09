import json
from pathlib import Path
p=Path('factors/miner_1_20301017_residualized_volnorm_drawdown_repair_speed_20_60obs.json')
d=json.loads(p.read_text())
metrics={
 'primary_horizon_days':20,'daily_paper_ic':0.047850,'daily_paper_icir':0.153304,
 'hit_ratio':0.5551,'ic_dates':1117,'coverage':0.6820,'mean_valid_instruments':10.23,
 'signal_cells':11614,'total_asset_date_cells':53220,'turnover':0.285277,
 'concentration_mean_cross_sectional_sd':2594.250454,
 'concentration_note':'Residual magnitudes can be large when the same-date control matrix is nearly collinear; downstream use must be rank based.',
 'decay':{'1d':{'ic':0.006726,'icir':0.020521,'hit_ratio':0.4974,'dates':1136},'5d':{'ic':0.013788,'icir':0.043055,'hit_ratio':0.5133,'dates':1132},'10d':{'ic':0.039401,'icir':0.122618,'hit_ratio':0.5448,'dates':1127},'20d':{'ic':0.047850,'icir':0.153304,'hit_ratio':0.5551,'dates':1117}},
 'max_abs_library_correlation':0.354384,'closest_library_factor':'volnorm_reversal_5obs','closest_library_factor_paired_cells':11460,'library_factors_compared':28,'library_missing_comparisons':['Two currently admitted signals were not reconstructable in this validation harness; this is revalidation evidence only, not a new admission novelty screen.']}
d['validation']['period']='2020-01-01 through 2030-12-25; usable IC observations 2026-01-01 through 2030-12-25'
d['validation']['metrics']=metrics
d['validation']['status']='EFFECTIVE'
d['validation']['regime_notes']={'2020-2021':'No usable IC dates after signal and forward-return availability filters.','2022-2023':'No usable IC dates after signal and forward-return availability filters.','2024-2025':'No usable IC dates after signal and forward-return availability filters.','2026-current':{'ic_dates':1117,'ic':0.047850,'icir':0.153304,'hit_ratio':0.5551},'revalidation_note':'20-day performance remains above shared IC and ICIR thresholds. This is a timely performance revalidation, not a new admission; its reconstructed novelty check covers 28 signals and is incomplete against the current 30-factor library.'}
d['last_validated']='2030-12-26'
d.setdefault('revalidation_history',[]).append({'validated_on':'2030-12-26','cutoff':'2030-12-25','status':'EFFECTIVE','primary_horizon_days':20,'daily_paper_ic':0.047850,'daily_paper_icir':0.153304,'ic_dates':1117,'coverage':0.6820,'turnover':0.285277,'max_abs_library_correlation':0.354384,'library_comparisons_reconstructed':28,'note':'Performance gates passed. Two-library-signal reconstruction gap means this correlation result cannot be used as complete admission evidence.'})
p.write_text(json.dumps(d,indent=2)+'\n')
print('updated',p)

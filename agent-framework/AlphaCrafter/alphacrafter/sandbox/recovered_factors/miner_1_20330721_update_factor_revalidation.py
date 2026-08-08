import json
p='factors/miner_1_20330303_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d.json'
d=json.load(open(p))
d['version']='revalidated_2033-07-21'
d['last_validated']='2033-07-21T09:30:00'
d['validation_timestamp']='2033-07-21T09:30:00'
v=d['validation']; v['period']='2020-01-01 through 2033-07-20; visibility-safe completed-close cutoff'; v['timestamp']='2033-07-21T09:30:00'; v['status']='EFFECTIVE'
m=v['metrics']; m.update({'primary_horizon_days':20,'daily_paper_ic':0.056319,'daily_paper_icir':0.164059,'ic_std':0.343284,'ic_standard_error':0.008237,'ic_hit_ratio':0.565343,'ic_dates':1737,'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.011514,'signal_cell_coverage':0.352758,'valid_cells':22319,'mean_rank_turnover':0.210453,'turnover_dates':1756,'library_factors_screened':34,'max_abs_library_correlation':0.40105,'max_abs_library_correlation_factor':'miner_3_residual_upside_volume_confirmation_60d','max_abs_library_correlation_common_cells':15278,'library_correlation_evidence_complete':True,'decay':{'1d':{'ic':0.014126,'icir':0.042825,'hit_ratio':0.506834,'ic_dates':1756},'5d':{'ic':0.032861,'icir':0.102235,'hit_ratio':0.539384,'ic_dates':1752},'10d':{'ic':0.039735,'icir':0.121098,'hit_ratio':0.5415,'ic_dates':1747},'20d':{'ic':0.056319,'icir':0.164059,'hit_ratio':0.565343,'ic_dates':1737}}})
r=v['regime_notes']; r['2027_onward_10d']={'ic_dates':1699,'ic':0.040595,'icir':0.123563,'hit_ratio':0.541495};r['interpretation']='Revalidation passes binding gates at 5, 10, and 20 days, strongest at 20 days. Full-library Spearman evidence is present and remains below the 0.5000 ceiling. The 2027-onward segment remains positive; the short 2025-26 segment is weak. Mean IC cross-section is 10.01 instruments, exceeding the eight-instrument minimum.'
b=d['benchmark_admission']['selected_metrics']; b.update({'ic':0.056319,'icir':0.164059,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.40105,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.056319*0.164059})
json.dump(d,open(p,'w'),indent=2);open(p,'a').write('\n')

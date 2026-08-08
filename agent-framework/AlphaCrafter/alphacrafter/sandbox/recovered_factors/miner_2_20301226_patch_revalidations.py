import json
DATE='2030-12-26'
common_period='2020-01-01 through 2030-12-25; completed-close visibility-safe cutoff'
items=[
('factors/miner_2_20300905_residual_commodity_crypto_dispersion_compression_loading_expansion_20_60d.json',{
'daily_paper_ic':.054300,'daily_paper_icir':.155679,'ic_std':.348794,'ic_standard_error':.010589,'ic_hit_ratio':.559447,'ic_dates':1085,'mean_valid_instruments':14.990783,'coverage':.400113,'mean_daily_rank_turnover':.124892,'turnover_dates':1104,'compression_nonzero_fraction':.323281,'max_abs_library_correlation':.369974,'most_correlated_library_factor':'miner_3_residual_return_dispersion_shock_sensitivity_expansion_20_60d','library_correlation_cells':21292,'library_factors_screened':27,'decay':{'1d':{'ic':-.008794,'icir':-.026053,'ic_dates':1104},'5d':{'ic':.010403,'icir':.032038,'ic_dates':1100},'10d':{'ic':.024391,'icir':.072986,'ic_dates':1095},'20d':{'ic':.054300,'icir':.155679,'ic_dates':1085}}},
'10-day IC: 2025-2026 / 2027 onward respectively: 0.031663 / 0.023925; ICIR 0.090876 / 0.071754. No aligned 2020-2024 observations after rolling-history requirements. Selected 20-day horizon remains above shared IC and ICIR gates; updated through 2030-12-25.'),
('factors/miner_2_20301003_residual_defensive_cyclical_dispersion_compression_loading_contraction_20_60d.json',{
'daily_paper_ic':.053286,'daily_paper_icir':.159078,'ic_std':.334971,'ic_standard_error':.010169,'ic_hit_ratio':.564977,'ic_dates':1085,'mean_valid_instruments_per_ic_date':14.990783,'signal_cell_coverage':.400113,'mean_rank_turnover':.116997,'turnover_dates':1104,'driver_nonzero_fraction':.323281,'max_abs_library_correlation':.096258,'max_abs_library_correlation_factor':'miner_2_downside_beta_improvement_120_20','max_abs_library_correlation_common_cells':17043,'library_factors_screened':29,'decay':{'1d':{'ic':-.002352,'icir':-.007157,'hit_ratio':.496377,'ic_dates':1104},'5d':{'ic':.015680,'icir':.047822,'hit_ratio':.533636,'ic_dates':1100},'10d':{'ic':.031630,'icir':.093241,'hit_ratio':.531507,'ic_dates':1095},'20d':{'ic':.053286,'icir':.159078,'hit_ratio':.564977,'ic_dates':1085}}},
'10-day IC: 2025-2026 / 2027 onward respectively: 0.159224 / 0.023446; ICIR 0.558727 / 0.068776. No aligned 2020-2024 observations after rolling-history requirements. Selected 20-day horizon remains above shared IC and ICIR gates; updated through 2030-12-25.')]
for path,vals,note in items:
 d=json.load(open(path)); d['version']=DATE; d['last_validated']=DATE; d['validation_timestamp']=DATE
 d['validation']['period']=common_period;d['validation']['timestamp']=DATE+'T09:30:00';d['validation']['status']='EFFECTIVE';d['validation']['regime_notes']=note;d['validation']['metrics'].update(vals)
 # Maintain an internally consistent admission summary, with the mandated correlation path.
 b=d.setdefault('benchmark_admission',{}).setdefault('selected_metrics',{})
 b.update({'ic':vals['daily_paper_ic'],'icir':vals['daily_paper_icir'],'metric_path':'validation.metrics','max_abs_library_correlation':vals['max_abs_library_correlation'],'correlation_path':'validation.metrics.max_abs_library_correlation','quality':abs(vals['daily_paper_ic'])*abs(vals['daily_paper_icir'])})
 json.dump(d,open(path,'w'),indent=2);print('updated',path)

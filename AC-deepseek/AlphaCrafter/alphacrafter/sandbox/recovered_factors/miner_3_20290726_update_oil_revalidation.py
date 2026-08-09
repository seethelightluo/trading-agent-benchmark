import json
from pathlib import Path
p=Path('factors/miner_3_20290419_oil_shock_transmission_beta_asymmetry_residual_30.json')
d=json.loads(p.read_text())
m=d['validation']['metrics']
m.update({
 'ic':0.049282,'icir':0.180555,'ic_horizon_days':20,'ic_dates':696,
 'ic_hit_ratio':0.5603,'mean_valid_instruments':13.01,'ic_standard_error':0.010346,
 'signal_cell_coverage':0.194923,'valid_signal_cells':9292,'mean_daily_rank_turnover':0.116460,
 'concentration':'Conditional 30-observation beta signal; each retained IC cross-section averages 13.01 of 15 instruments, above the eight-instrument requirement.',
 'decay':{'1d':{'ic':0.007466,'icir':0.025594,'dates':713,'hit_ratio':0.5077},'5d':{'ic':0.025636,'icir':0.090655,'dates':709,'hit_ratio':0.5289},'10d':{'ic':0.039269,'icir':0.145388,'dates':704,'hit_ratio':0.5611},'20d':{'ic':0.049282,'icir':0.180555,'dates':696,'hit_ratio':0.5603}},
 'max_abs_library_correlation':0.158337,'closest_library_factor':'dispersion_conditioned_market_capture_residual_30','common_signal_observations_closest':9292,'admission_quality_score':0.0089008})
d['validation'].update({'period':'visible daily panel through 2029-07-25; forward-return panel aligned without look-ahead through the final available 20-day outcome','timestamp':'2029-07-26T00:00:00Z','status':'EFFECTIVE','regime_notes':'Full-sample 20-day IC and ICIR remain above the binding admission gates. 2026-27: 364 dates, IC 0.070590, ICIR 0.259842, hit 62.36%. 2028 through 2029-07-25: 340 dates, IC 0.005737, ICIR 0.021672, hit 49.41%; this recent segment no longer independently clears either gate. Full evidence against all 21 admitted library signals has maximum absolute Spearman correlation 0.158337, below 0.500000, versus dispersion_conditioned_market_capture_residual_30 over 9,292 common cells. Keep effective under the full-sample contract, but reduce confidence and revalidate again within one quarter.'})
d['last_validated']='2029-07-26T00:00:00Z'; d['revalidation_due']='2029-10-26'
d['benchmark_admission']['selected_metrics'].update({'ic':0.049282,'icir':0.180555,'max_abs_library_correlation':0.158337,'quality':0.0089008})
p.write_text(json.dumps(d,indent=2)+'\n')

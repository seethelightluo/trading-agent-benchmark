import json
p='factors/miner_3_20270325_downside_beta_asymmetry_30.json'
d=json.load(open(p))
d['version']='2028-09-07 revalidation'
d['last_validated']='2028-09-07T00:00:00Z'
d['revalidation_due']='2028-12-07'
d['validation']['period']='aligned available panel through 2028-09-06; forward labels restricted to the visible research panel'
m=d['validation']['metrics']
m.update({'ic':0.0256410459,'icir':0.0901315656,'ic_horizon_days':10,'ic_dates':519,'ic_hit_ratio':0.5452793834,'mean_valid_instruments':14.9537572254,'ic_standard_error':0.0124874897,'signal_cell_coverage':0.278132,'mean_daily_rank_turnover':0.085216,'concentration':'Broad signal: mean 14.95 valid instruments out of 15 per IC date.','decay':{'1d':{'ic':0.009590296,'icir':0.031836912,'dates':528,'hit_ratio':0.4962121212},'5d':{'ic':0.0189687643,'icir':0.0665611774,'dates':524,'hit_ratio':0.5400763359},'10d':{'ic':0.0256410459,'icir':0.0901315656,'dates':519,'hit_ratio':0.5452793834},'20d':{'ic':-0.007756656,'icir':-0.025276918,'dates':509,'hit_ratio':0.5029469538}},'max_abs_library_correlation':0.103239,'closest_library_factor':'vix_beta_residual_peer20','common_signal_observations_closest':6877,'library_factors_tested':18,'valid_signal_cells':12299,'admission_quality_score':0.002310})
d['validation']['regime_notes']='The decision-aligned 10-day aggregate result remains above the binding gates (519 IC dates; IC 0.025641, ICIR 0.090132, hit 54.53%) and the mandatory library-correlation gate remains clear (maximum absolute Spearman rho 0.103239 with vix_beta_residual_peer20; 6,877 common cells). However, performance has materially decayed: 2026-27 has IC 0.047448 and ICIR 0.178110 (370 dates), whereas 2028 YTD is negative, IC -0.028511 and ICIR -0.089195 (149 dates; hit 46.98%). Retained because full visible-sample binding gates pass, but flagged for close monitoring and no increased ensemble allocation; next revalidation due 2028-12-07.'
d['benchmark_admission']['selected_metrics'].update({'ic':0.0256410459,'icir':0.0901315656,'max_abs_library_correlation':0.103239,'quality':0.002310})
open(p,'w').write(json.dumps(d,indent=2)+'\n')

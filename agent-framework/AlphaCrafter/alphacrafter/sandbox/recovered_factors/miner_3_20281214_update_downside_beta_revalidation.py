import json
p='factors/miner_3_20270325_downside_beta_asymmetry_30.json'
with open(p,encoding='utf-8') as f: d=json.load(f)
m=d['validation']['metrics']
m.update({'ic':0.0303858259,'icir':0.1040211861,'ic_horizon_days':10,'ic_dates':587,'ic_hit_ratio':0.5417376491,'mean_valid_instruments':14.9591141397,'ic_standard_error':0.0120567483,'signal_cell_coverage':0.294213,'mean_daily_rank_turnover':0.085827,'concentration':'Broad signal: mean 14.96 valid instruments out of 15 per IC date.','decay':{'1d':{'ic':0.0112603327,'icir':0.0372382908,'dates':596,'hit_ratio':0.4983221477},'5d':{'ic':0.0269007500,'icir':0.0947054548,'dates':592,'hit_ratio':0.5489864865},'10d':{'ic':0.0303858259,'icir':0.1040211861,'dates':587,'hit_ratio':0.5417376491},'20d':{'ic':0.0043228911,'icir':0.0139041206,'dates':577,'hit_ratio':0.5008665511}},'max_abs_library_correlation':0.103792,'closest_library_factor':'vix_beta_residual_peer20','common_signal_observations_closest':7761,'admission_quality_score':0.003160773,'library_factors_tested':18,'valid_signal_cells':13319})
d['version']='2028-12-14 revalidation';d['last_validated']='2028-12-14T00:00:00Z';d['revalidation_due']='2029-03-14'
d['validation']['period']='aligned available panel through 2028-12-13; forward labels restricted to the visible research panel'
d['validation']['status']='EFFECTIVE'
d['validation']['regime_notes']='Decision-aligned 10-day aggregate evidence clears the binding gates: 587 IC dates, IC 0.030386, ICIR 0.104021, hit 54.17%; 5-day results also pass (IC 0.026901, ICIR 0.094705). Signal breadth is 14.96 of 15 instruments per IC date and maximum absolute Spearman correlation against all 18 admitted comparison signals is 0.103792 (VIX beta residual; 7,761 common cells). The 2026-27 regime remained strong (370 dates, IC 0.047448, ICIR 0.178110), but 2028 YTD has decayed to nearly neutral (217 dates, IC 0.001293, ICIR 0.003918; hit 48.39%). Retained on full-sample contract compliance, under monitoring; do not increase ensemble allocation solely on this factor.'
d['benchmark_admission']['selected_metrics'].update({'ic':0.0303858259,'icir':0.1040211861,'max_abs_library_correlation':0.103792,'quality':0.003160773})
with open(p,'w',encoding='utf-8') as f: json.dump(d,f,indent=2);f.write('\n')
print('updated',p,d['validation']['status'],d['last_validated'])

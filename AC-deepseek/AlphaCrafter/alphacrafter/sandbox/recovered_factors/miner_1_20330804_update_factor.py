import json
p='factors/miner_1_20330120_trend_orthogonal_overnight_intraday_resilience_40.json'
d=json.load(open(p))
m=d['validation']['metrics']
m.update({'coverage_cells':23231,'total_cells':63420,'coverage':0.366304,'turnover_rank_change':0.085517,'mean_cross_sectional_std':0.001916,'max_abs_library_correlation':0.349672,'most_correlated_library_factor':'moderate_downside_peer_relative_capture_60','correlation_evidence_cells':472,'library_factors_audited':37,'horizons':{'1d':{'ic':0.02702,'icir':0.086308,'hit_ratio':0.534502,'ic_dates':1826,'mean_valid_instruments':12.716,'min_valid_instruments':12},'5d':{'ic':0.040569,'icir':0.131635,'hit_ratio':0.553238,'ic_dates':1822,'mean_valid_instruments':12.717,'min_valid_instruments':12},'10d':{'ic':0.043966,'icir':0.145413,'hit_ratio':0.55366,'ic_dates':1817,'mean_valid_instruments':12.719,'min_valid_instruments':12},'20d':{'ic':0.049368,'icir':0.16575,'hit_ratio':0.572219,'ic_dates':1807,'mean_valid_instruments':12.723,'min_valid_instruments':12}},'decay':'Positive IC rises from 1 to 20 sessions; 20-session horizon selected, though latest 180-calendar-day window has reversed negative.'})
d['validation']['period']='2020-01-01 to 2033-08-03 (data visible at 2033-08-04 decision time)'
d['validation']['status']='EFFECTIVE'
d['validation']['regime_notes']={'2023_2026_20d':{'ic_dates':108,'ic':0.069827,'icir':0.309478,'hit_ratio':0.657407},'2027_2033_08_03_20d':{'ic_dates':1699,'ic':0.048067,'icir':0.159238,'hit_ratio':0.566804},'recent_180_calendar_days_20d':{'ic_dates':109,'ic':-0.063705,'icir':-0.261341,'hit_ratio':0.431193},'note':'Full-history and both long subperiods remain admission-compliant with at least 12 instruments per IC date. The recent window is negative and requires an early revalidation; this is a performance-drift warning, not a full-sample admission failure.'}
d['last_validated']='2033-08-04'
d['benchmark_admission']['selected_metrics']={'ic':0.049368,'icir':0.16575,'metric_path':'validation.metrics.horizons.20d','max_abs_library_correlation':0.349672,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.008182236}
open(p,'w').write(json.dumps(d,indent=2)+'\n')

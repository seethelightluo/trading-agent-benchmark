"""Deprecate drift-failed overnight/intraday resilience factor after revalidation."""
import json, os
src='factors/miner_1_20330120_trend_orthogonal_overnight_intraday_resilience_40.json'
dst='factors/miner_1_20330120_trend_orthogonal_overnight_intraday_resilience_40_deprecated.json'
with open(src,encoding='utf-8') as f: x=json.load(f)
x['version']='2033-10-13 deprecation after accelerated revalidation'
x['last_validated']='2033-10-13'
x['validation']['status']='DEPRECATED'
x['validation']['period']='2020-01-01 to 2033-10-12 (data visible at 2033-10-13 decision time)'
x['validation']['metrics'].update({
 'coverage_cells':23831,'total_cells':64170,'coverage':0.371373,
 'turnover_rank_change':0.085406,'mean_cross_sectional_std':0.001915,
 'max_abs_library_correlation':0.349672,
 'most_correlated_library_factor':'moderate_downside_peer_relative_capture_60',
 'correlation_evidence_cells':472,'library_factors_audited':37,
 'horizons':{
  '1d':{'ic':0.027358,'icir':0.087158,'hit_ratio':0.534115,'ic_dates':1876,'mean_valid_instruments':12.697,'min_valid_instruments':12},
  '5d':{'ic':0.041994,'icir':0.136026,'hit_ratio':0.555556,'ic_dates':1872,'mean_valid_instruments':12.698,'min_valid_instruments':12},
  '10d':{'ic':0.046292,'icir':0.152786,'hit_ratio':0.555972,'ic_dates':1867,'mean_valid_instruments':12.7,'min_valid_instruments':12},
  '20d':{'ic':0.045295,'icir':0.149654,'hit_ratio':0.565428,'ic_dates':1857,'mean_valid_instruments':12.704,'min_valid_instruments':12}},
 'decay':'Full-history IC remains positive through 20 sessions, but selected 20-session recent-window IC and ICIR are materially negative; deprecated for performance drift.'})
x['validation']['regime_notes']={
 '2023_2026_20d':{'ic_dates':108,'ic':0.069827,'icir':0.309478,'hit_ratio':0.657407},
 '2027_2033_10_12_20d':{'ic_dates':1749,'ic':0.04378,'icir':0.142711,'hit_ratio':0.559748},
 'recent_180_calendar_days_10d':{'ic_dates':119,'ic':0.030812,'icir':0.096533,'hit_ratio':0.537815},
 'recent_180_calendar_days_20d':{'ic_dates':109,'ic':-0.092033,'icir':-0.271336,'hit_ratio':0.385321},
 'note':'Despite full-sample gates and correlation diversification, the selected 20-session horizon has a negative recent ICIR. Under the required drift policy, the factor is deprecated.'}
x['benchmark_admission']['selected_metrics']={
 'ic':-0.092033,'icir':-0.271336,
 'metric_path':'validation.regime_notes.recent_180_calendar_days_20d',
 'max_abs_library_correlation':0.349672,
 'correlation_path':'validation.metrics.max_abs_library_correlation',
 'quality':0.01344,
 'revalidation_outcome':'FAILED: selected 20-session recent 180-calendar-day ICIR is negative; deprecated for drift.'}
x['deprecation_reason']='2033-10-13 accelerated revalidation: selected 20-session recent-window IC=-0.092033 and ICIR=-0.271336.'
with open(dst,'w',encoding='utf-8') as f: json.dump(x,f,indent=2);f.write('\n')
os.remove(src)
print('DEPRECATED',dst)

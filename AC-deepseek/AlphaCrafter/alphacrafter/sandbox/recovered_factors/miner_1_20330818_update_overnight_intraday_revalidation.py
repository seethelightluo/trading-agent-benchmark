"""Update persisted revalidation evidence for one existing factor; cutoff 2033-08-17."""
import json
p='factors/miner_1_20330120_trend_orthogonal_overnight_intraday_resilience_40.json'
with open(p,encoding='utf-8') as f: d=json.load(f)
m=d['validation']['metrics']
m.update({
 'coverage_cells':23351,'total_cells':63570,'coverage':0.367327,
 'turnover_rank_change':0.085470,'mean_cross_sectional_std':0.001918,
 'max_abs_library_correlation':0.349672,
 'most_correlated_library_factor':'moderate_downside_peer_relative_capture_60',
 'correlation_evidence_cells':472,'library_factors_audited':37,
 'horizons':{
  '1d':{'ic':0.027542,'icir':0.087927,'hit_ratio':0.535403,'ic_dates':1836,'mean_valid_instruments':12.712,'min_valid_instruments':12},
  '5d':{'ic':0.041792,'icir':0.135603,'hit_ratio':0.555131,'ic_dates':1832,'mean_valid_instruments':12.713,'min_valid_instruments':12},
  '10d':{'ic':0.043757,'icir':0.144598,'hit_ratio':0.552819,'ic_dates':1827,'mean_valid_instruments':12.715,'min_valid_instruments':12},
  '20d':{'ic':0.045670,'icir':0.151628,'hit_ratio':0.569070,'ic_dates':1817,'mean_valid_instruments':12.719,'min_valid_instruments':12}},
 'decay':'Positive IC increases from 1 to 20 sessions, with 20 sessions selected. The most recent 180-calendar-day diagnostic is materially negative and requires continued accelerated monitoring.'})
d['validation']['period']='2020-01-01 to 2033-08-17 (data visible at 2033-08-18 decision time)'
d['validation']['status']='EFFECTIVE'
d['validation']['regime_notes']={
 '2023_2026_20d':{'ic_dates':108,'ic':0.069827,'icir':0.309478,'hit_ratio':0.657407},
 '2027_2033_08_17_20d':{'ic_dates':1709,'ic':0.044144,'icir':0.144575,'hit_ratio':0.563487},
 'recent_180_calendar_days_20d':{'ic_dates':109,'ic':-0.105167,'icir':-0.368119,'hit_ratio':0.403670},
 'note':'Full-history and long regimes pass the shared gates (at least 12 instruments per IC date), and library correlation remains below 0.5. Recent 180-calendar-day performance has worsened versus the prior review and is negative; factor remains EFFECTIVE on full-sample evidence but is on drift watch and must be revalidated again before the normal quarterly schedule.'}
d['last_validated']='2033-08-18'
d['version']='2033-08-18 revalidation'
with open(p,'w',encoding='utf-8') as f: json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n')
print('UPDATED',p,'STATUS',d['validation']['status'],'LAST_VALIDATED',d['last_validated'])
PYTHON_PLACEHOLDER

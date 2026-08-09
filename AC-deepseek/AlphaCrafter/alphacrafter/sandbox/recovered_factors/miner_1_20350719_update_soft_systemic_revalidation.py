import json
p='factors/miner_1_20350524_library_orthogonal_soft_systemic_weakness_relative_downside_persistence_10_60.json'
with open(p,encoding='utf-8') as f: x=json.load(f)
v=x['validation']; v['period']='2020-01-01 through 2035-07-18 (visible completed-session cutoff)'; v['status']='EFFECTIVE'
v['metrics']={
 'ic':-0.023459,'icir':-0.087531,'forward_horizon_sessions':10,'ic_hit_ratio':0.461505,'ic_dates':2325,
 'mean_instruments_per_ic_date':14.901,'min_instruments_per_ic_date':10,'coverage':0.489588,
 'coverage_cells':34795,'total_cells':71070,'turnover_mean_abs_daily_percentile_rank_change':0.141209,
 'cross_sectional_dispersion':0.0972,
 'decay':{'1_sessions':{'ic':-0.009723,'icir':-0.03518,'dates':2334},'5_sessions':{'ic':-0.017722,'icir':-0.064825,'dates':2330},'10_sessions':{'ic':-0.023459,'icir':-0.087531,'dates':2325},'20_sessions':{'ic':-0.0219,'icir':-0.08162,'dates':2315}},
 'max_abs_library_correlation':0.323534,'most_correlated_library_factor':'peer_relative_downside_volatility_compression_10_40','library_correlation_evidence_cells':34795,'audited_library_factor_count':37}
v['regime_notes']={'2023_2026_10d':{'dates':106,'ic':-0.094992,'icir':-0.386203,'hit_ratio':0.330189},'2027_2035_07_18_10d':{'dates':2219,'ic':-0.020042,'icir':-0.074619,'hit_ratio':0.467778},'recent_180_calendar_days_10d':{'dates':119,'ic':-0.079005,'icir':-0.293553,'hit_ratio':0.386555}}
v['admission_note']='Scheduled revalidation passes the binding same-horizon absolute IC and ICIR gates at 10 sessions; negative IC specifies inverse implementation. The latest 180-calendar-day performance is materially weak, so retain only with heightened monitoring and revalidate again on the next research cycle.'
x['last_validated']='2035-07-19'
x['benchmark_admission']['selected_metrics']={'ic':-0.023459,'icir':-0.087531,'metric_path':'validation.metrics','max_abs_library_correlation':0.323534,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.002053}
with open(p,'w',encoding='utf-8') as f: json.dump(x,f,indent=2);f.write('\n')
print('UPDATED',p,x['validation']['status'],x['last_validated'])

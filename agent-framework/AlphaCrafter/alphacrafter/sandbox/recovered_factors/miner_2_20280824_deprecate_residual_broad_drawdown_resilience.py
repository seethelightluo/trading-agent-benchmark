import json,pathlib
p=pathlib.Path('factors/miner_2_20280323_residual_broad_drawdown_resilience_60d.json')
d=json.loads(p.read_text())
v=d['validation']; m=v['metrics']
v['period']='2020-01-01 through 2028-08-23; eligible IC observations begin in 2025 because of rolling and aligned-calendar requirements'
v['timestamp']='2028-08-24T09:30:00'; v['status']='DEPRECATED'
m.update({'primary_horizon_days':20,'daily_paper_ic':-0.001047,'daily_paper_icir':-0.002856,'ic_std':0.366684,'ic_standard_error':0.016789,'ic_hit_ratio':0.494759,'ic_dates':477,'universe_instruments':15,'mean_valid_instruments_per_ic_date':14.979036,'signal_cell_coverage':0.276242,'mean_rank_turnover':0.048029,'turnover_dates':496,'latest_valid_instruments':15,'stress_session_frequency':0.702859,'max_abs_library_correlation':0.488674,'max_abs_library_correlation_factor':'miner_3_residual_lower_partial_moment_60d','library_signal_cells_compared':10450,'decay':{'1d':{'ic':0.010051,'icir':0.030569,'hit_ratio':0.516129,'dates':496,'mean_valid_instruments':14.979839},'5d':{'ic':-0.007497,'icir':-0.022421,'hit_ratio':0.502033,'dates':492,'mean_valid_instruments':14.979675},'10d':{'ic':0.005919,'icir':0.016909,'hit_ratio':0.50308,'dates':487,'mean_valid_instruments':14.979466},'20d':{'ic':-0.001047,'icir':-0.002856,'hit_ratio':0.494759,'dates':477,'mean_valid_instruments':14.979036}}})
v['regime_notes']={'2025_2026':{'ic_20d':-0.190844,'icir_20d':-0.762991,'hit_ratio':0.279412,'ic_dates':68},'2027_2028_08_23':{'ic_20d':0.030508,'icir':0.081655,'hit_ratio':0.530562,'ic_dates':409},'interpretation':'The historical negative 20-session relationship has decayed and reversed after 2026. Aggregate 20-day IC and ICIR no longer clear admission gates, so the factor is deprecated despite correlation remaining narrowly below the 0.5 limit.'}
d['last_validated']='2028-08-24T09:30:00'
d['benchmark_admission']['selected_metrics']={'ic':-0.001047,'icir':-0.002856,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.488674,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.000002989032}
d['deprecation_reason']='Failed scheduled re-validation: no tested horizon has both absolute paper IC >= 0.007 and absolute ICIR >= 0.084; primary 20d relation is effectively zero.'
out=p.with_name(p.stem+'_deprecated.json');out.write_text(json.dumps(d,indent=2)+'\n');p.unlink()
print(out)

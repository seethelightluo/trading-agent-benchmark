import json,pathlib
old=pathlib.Path('factors/miner_1_20280127_residualized_inflation_basket_correlation_decoupling_60_20.json')
d=json.loads(old.read_text())
m=d['validation']['metrics']
m.update({'daily_paper_ic':0.018191,'daily_paper_icir':0.056845,'ic_std':0.320017,'ic_standard_error':0.01394,'ic_hit_ratio':0.472486,'ic_dates':527,'mean_valid_instruments':10.216319,'coverage':0.122767,'rank_turnover':0.129767,'turnover_dates':536,'concentration':'continuous residual score; mean 10.22 valid names on eligible dates','max_abs_library_correlation':0.364101,'most_correlated_library_factor':'miner_1_market_beta_contraction_60_20','library_correlation_cells':5484,'library_factors_compared':21,'decay':{'1d':{'ic':-0.001762,'icir':-0.005832,'ic_dates':536,'hit_ratio':0.503731},'5d':{'ic':0.034412,'icir':0.112318,'ic_dates':532,'hit_ratio':0.511278},'10d':{'ic':0.018191,'icir':0.056845,'ic_dates':527,'hit_ratio':0.472486},'20d':{'ic':-0.005846,'icir':-0.018488,'ic_dates':517,'hit_ratio':0.466151}}})
d['validation']['period']='2020-01-01 through 2028-10-18; point-in-time cutoff 2028-10-18'
d['validation']['status']='DEPRECATED'
d['validation']['regime_notes']='All 527 eligible 10-session IC observations remain in the available 2026-28 synthetic segment. The formerly selected 10-session relationship decayed to IC 0.018191 and ICIR 0.056845, below the 0.084 benchmark ICIR gate. Although the 5-session result still passes absolute admission gates, its isolated short-horizon efficacy and sign reversal at 20 sessions are not suitable for the 10-session implementation cadence.'
d['validation']['admission_basis']='DEPRECATED on revalidation: selected 10-session ICIR=0.056845 < 0.084000. The 5-session IC/ICIR remains 0.034412/0.112318 and library correlation remains independent (0.364101), but the active 10-session relationship no longer meets the stability gate.'
d['validation']['revalidation_history'].append({'date':'2028-10-19','cutoff':'2028-10-18','status':'DEPRECATED','selected_horizon_sessions':10,'ic':0.018191,'icir':0.056845,'max_abs_library_correlation':0.364101,'library_factors_compared':21,'note':'5d passes but operational 10d ICIR failed; 20d reverses sign.'})
d['last_validated']='2028-10-19';d['version']='2028-10-19'
d['benchmark_admission']['selected_metrics']={'ic':0.018191,'icir':0.056845,'metric_path':'validation.metrics.decay.10d','max_abs_library_correlation':0.364101,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.001033}
new=old.with_name(old.stem+'_deprecated.json');new.write_text(json.dumps(d,indent=2)+'\n');old.unlink()
print(new)

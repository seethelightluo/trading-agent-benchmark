import json
p='factors/miner_3_20300530_conditional_dxy_impulse_exposure_5v40obs.json'
x=json.load(open(p))
m=x['validation']['metrics']
m.update({'primary_horizon_days':20,'daily_paper_ic':0.027939492996648684,'daily_paper_icir':0.09347339711335427,'ic_hit_ratio':0.5198156682027649,'ic_standard_error':0.009074352071783484,'ic_dates':1085,'universe_instruments':15,'mean_valid_instruments_per_ic_date':14.990783410138249,'signal_cell_coverage':0.4049600912200685,'mean_instruments_per_panel_date':6.074401368301026,'rank_stability_1d':0.5714990961559453,'turnover_proxy_rank_stability_1d':0.5714990961559453,'max_abs_library_correlation':0.4094929270931646,'most_correlated_library_factor':'miner_3_conditional_usdjpy_impulse_exposure_10v50obs','max_abs_library_correlation_common_signal_cells':20685,'library_correlation_evidence_complete':True,'library_factors_compared':26,'decay_ic':{'1d':-0.006624019855658785,'5d':-0.009259388881044004,'10d':0.014161370110933132,'20d':0.027939492996648684},'decay_icir':{'1d':-0.022785380738161012,'5d':-0.031874071609538826,'10d':0.0487188494828019,'20d':0.09347339711335427},'decay_ic_dates':{'1d':1104,'5d':1100,'10d':1095,'20d':1085}})
x['validation'].update({'period':'2020-01-01 through 2030-10-30','timestamp':'2030-10-31','status':'EFFECTIVE','regime_notes':'Primary 20-observation IC: 2024-2026, 106 dates, -0.06224 / ICIR -0.24296; 2027-2030, 979 dates, 0.03770 / ICIR 0.12498. Full-history gate remains satisfied, but the opposed early regime and modest margin above the ICIR threshold require modest use and quarterly re-validation.'})
x['last_validated']='2030-10-31'
x['benchmark_admission']['selected_metrics'].update({'ic':0.027939492996648684,'icir':0.09347339711335427,'max_abs_library_correlation':0.4094929270931646,'quality':0.002611579351828399})
open(p,'w').write(json.dumps(x,indent=2)+'\n')
print('UPDATED',p,x['validation']['status'],x['last_validated'])

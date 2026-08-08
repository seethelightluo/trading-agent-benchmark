import json
p='factors/miner_3_20300613_conditional_usdcny_impulse_exposure_10v50obs.json'
j=json.load(open(p))
m={
 'primary_horizon_days':20,'daily_paper_ic':-0.04214243892694287,'daily_paper_icir':-0.14493853329980283,'ic_hit_ratio':0.4422518862449216,'ic_standard_error':0.007004758472661779,'ic_dates':1723,'universe_instruments':15,'mean_valid_instruments_per_ic_date':14.99651770168311,'signal_cell_coverage':0.49665702346512375,'mean_instruments_per_panel_date':7.449855351976856,'rank_stability_1d':0.7059781011415583,'turnover_proxy_rank_stability_1d':0.29402189885844165,'max_abs_library_correlation':0.12407317799057649,'most_correlated_library_factor':'downside_concentration_continuation_10v40obs','max_abs_library_correlation_common_signal_cells':9752,'library_correlation_evidence_complete':True,'library_factors_compared':29,
 'decay_ic':{'1d':-0.013100801263259463,'5d':-0.027048147547112666,'10d':-0.03630439595986497,'20d':-0.04214243892694287},
 'decay_icir':{'1d':-0.0448658802467678,'5d':-0.0919307475026544,'10d':-0.12345996424186512,'20d':-0.14493853329980283},
 'decay_ic_dates':{'1d':1742,'5d':1738,'10d':1733,'20d':1723}}
j['version']='2033-04-14';j['last_validated']='2033-04-14';j['validation'].update({'period':'2020-01-01 through 2033-04-13','timestamp':'2033-04-14','status':'EFFECTIVE','metrics':m,'regime_notes':'Primary 20-observation horizon clears the binding absolute gates (|IC| 0.04214, |ICIR| 0.14494); negative orientation remains intentionally inverted downstream. Regimes: 2024-2026 (104 dates) IC -0.02102 / ICIR -0.08760; 2027-2030 (1,043) -0.05463 / -0.18218; 2031-2032 (523) -0.00910 / -0.03239; 2033 YTD (53) -0.16401 / -0.65781. The severe, albeit short, 2033 deterioration is a drift warning. Full-history evidence and complete 29-factor correlation evidence remain admissible, so retain with elevated monitoring and revalidate next cycle.'})
j['benchmark_admission']['selected_metrics'].update({'ic':m['daily_paper_ic'],'icir':m['daily_paper_icir'],'max_abs_library_correlation':m['max_abs_library_correlation'],'quality':abs(m['daily_paper_ic']*m['daily_paper_icir'])})
with open(p,'w') as f:json.dump(j,f,indent=2);f.write('\n')
print('updated',p,'quality',j['benchmark_admission']['selected_metrics']['quality'])

import json
p='factors/miner_3_20300613_conditional_usdcny_impulse_exposure_10v50obs.json'
j=json.load(open(p))
m=j['validation']['metrics']
m.update({
 'primary_horizon_days':20,'daily_paper_ic':-0.05508855328815336,'daily_paper_icir':-0.18507463968271357,'ic_hit_ratio':0.43557772236076475,'ic_standard_error':0.008581864261556138,'ic_dates':1203,'universe_instruments':15,'mean_valid_instruments_per_ic_date':14.99501246882793,'signal_cell_coverage':0.42451304667401685,'mean_instruments_per_panel_date':6.367695700110254,'rank_stability_1d':0.7044230013459626,'turnover_proxy_rank_stability_1d':0.7044230013459626,
 'max_abs_library_correlation':0.12407317799057649,'most_correlated_library_factor':'downside_concentration_continuation_10v40obs','max_abs_library_correlation_common_signal_cells':9752,'library_correlation_evidence_complete':True,'library_factors_compared':28,
 'decay_ic':{'1d':-0.01228414085904844,'5d':-0.029107824146768272,'10d':-0.04742860907982738,'20d':-0.05508855328815336},
 'decay_icir':{'1d':-0.04263158913074652,'5d':-0.09945363857321231,'10d':-0.1598940251743774,'20d':-0.18507463968271357},
 'decay_ic_dates':{'1d':1222,'5d':1218,'10d':1213,'20d':1203}
})
j['version']='2031-04-17';j['last_validated']='2031-04-17';j['validation'].update({'period':'2020-01-01 through 2031-04-16','timestamp':'2031-04-17','status':'EFFECTIVE','regime_notes':'Primary 20-observation horizon: 2024-2026, 104 IC dates, IC -0.02102 and ICIR -0.08760; 2027-2030, 1,043 dates, IC -0.05463 and ICIR -0.18218; 2031 YTD, 56 dates, IC -0.12695 and ICIR -0.37025. All synchronized regimes retain the same negative direction, with recent strength improving.'})
j['benchmark_admission']['selected_metrics'].update({'ic':-0.05508855328815336,'icir':-0.18507463968271357,'max_abs_library_correlation':0.12407317799057649,'quality':0.010194607578711186})
open(p,'w').write(json.dumps(j,indent=2)+'\n')

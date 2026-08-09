import json
p='factors/miner_3_20300711_conditional_usdjpy_impulse_exposure_10v50obs.json'
j=json.load(open(p))
j['validation']['metrics']={'primary_horizon_observations':10,'daily_paper_ic':0.03254296004293454,'daily_paper_icir':0.11515707815040442,'ic_hit_ratio':0.5430410297666934,'ic_standard_error':0.008015504301791182,'ic_dates':1243,'mean_valid_instruments':14.995172968624296,'panel_dates':1252,'signal_coverage':0.4292327319117915,'mean_active_names_per_date':6.438490978339093,'rank_stability_1d':None,'decay':{'1':{'daily_paper_ic':0.004979662834279914,'daily_paper_icir':0.016871240964549627,'ic_dates':1252},'5':{'daily_paper_ic':0.02514593715033163,'daily_paper_icir':0.08567993967206848,'ic_dates':1248},'10':{'daily_paper_ic':0.03254296004293454,'daily_paper_icir':0.11515707815040442,'ic_dates':1243},'20':{'daily_paper_ic':0.0224467140963921,'daily_paper_icir':0.07425424002077056,'ic_dates':1233}},'max_abs_library_correlation':0.11131026967092263,'most_correlated_library_factor':'downside_concentration_continuation_10v40obs','library_correlation_common_signal_cells':9752,'library_correlation_evidence_complete':True}
j['validation']['period']='2020-01-01 through 2031-05-28';j['validation']['status']='EFFECTIVE';j['validation']['regime_notes']='10-observation IC: 2024–2026 -0.10760 (ICIR -0.53021; 104 dates), 2027–2030 0.04444 (0.15259; 1,043), 2031 YTD 0.05512 (0.25806; 96). Historical early-regime weakness remains material, while most-recent performance is positive.'
j['last_validated']='2031-05-29'
open(p,'w').write(json.dumps(j,indent=2)+'\n')
print('updated',p)

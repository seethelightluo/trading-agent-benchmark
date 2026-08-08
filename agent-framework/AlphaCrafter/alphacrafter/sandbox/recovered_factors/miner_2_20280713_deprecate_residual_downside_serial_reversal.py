import json,pathlib
p=pathlib.Path('factors/miner_2_20270729_residual_downside_serial_reversal_60d.json')
d=json.loads(p.read_text())
d['version']='2028-07-13'
d['last_validated']='2028-07-13T15:00:00'
d['validation']['status']='DEPRECATED'
d['validation']['period']='2020-01-01 through 2028-07-12; factor availability and eligible IC observations begin in 2025'
d['validation']['metrics']={'selected_horizon_days':20,'daily_paper_ic':-0.023162,'daily_paper_icir':-0.070131,'ic_hit_ratio':0.471655,'ic_dates':441,'mean_valid_instruments':10.609977,'coverage':0.220266,'latest_valid_instruments':10,'mean_daily_rank_turnover':0.037665,'turnover_dates':460,'max_abs_library_correlation':0.128872,'most_correlated_library_factor':'miner_1_breadth_recovery_capture_60d','library_correlation_evidence':'Exhaustive 27-signal reconstructed admitted library through 2028-07-12: maximum abs(rho)=0.128872 against miner_1_breadth_recovery_capture_60d over 9,608 common cells. Diversification gate passes, but ICIR gate fails.','decay':{'1d':{'ic':-0.008816,'icir':-0.029494,'ic_dates':460},'5d':{'ic':-0.010483,'icir':-0.034906,'ic_dates':456},'10d':{'ic':-0.017327,'icir':-0.056525,'ic_dates':451},'20d':{'ic':-0.023162,'icir':-0.070131,'ic_dates':441}},'regime_20d':{'2025_2026':{'ic':0.136035,'icir':0.456648,'hit_ratio':0.645161,'dates':62},'2027_to_2028_07_12':{'ic':-0.049204,'icir':-0.149859,'hit_ratio':0.443272,'dates':379}}}
d['validation']['regime_notes']='Failed scheduled re-validation: full-sample 20-day ICIR is -0.070131, below the required absolute 0.084000, and post-2027 relation reversed (IC -0.049204, ICIR -0.149859). It is deprecated despite passing library-diversification screen.'
out=p.with_name(p.stem+'_deprecated.json')
out.write_text(json.dumps(d,indent=2)+'\n')
p.unlink()
print(out)

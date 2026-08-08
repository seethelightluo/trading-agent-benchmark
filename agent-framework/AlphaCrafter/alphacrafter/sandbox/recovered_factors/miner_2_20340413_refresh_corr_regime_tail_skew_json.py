import json
p='factors/miner_2_20331124_corr_regime_inverse_tail_skew_20v60obs.json'
j=json.load(open(p))
m=j['validation']['metrics']
m.update({'daily_paper_ic':0.033873099922482636,'daily_paper_icir':0.09335781866484005,'ic_hit_ratio':0.5294924554183813,'ic_standard_error':0.013438180907934296,'ic_dates':729,'mean_valid_instruments':9.156378600823045,'coverage':0.23998790078644888,'mean_available_instruments':3.5998185117967334,'mean_rank_stability_1d':0.8910293264048206,'implied_rank_turnover':0.10897067359517942,'decay':{'1d':{'ic':-0.0025749056613254143,'icir':-0.007358989498680819,'dates':729},'5d':{'ic':0.01100890421878076,'icir':0.03155898123580146,'dates':729},'10d':{'ic':-0.004330329021687048,'icir':-0.012523127116491341,'dates':729},'20d':{'ic':0.033873099922482636,'icir':0.09335781866484005,'dates':729}},'max_abs_library_correlation':0.4495968305164807,'most_correlated_library_factor':'miner_2_standardized_jump_asymmetry_20v40obs','library_correlation_evidence_complete':True})
j['version']='2034-04-13';j['last_validated']='2034-04-13';j['signal_artifact']='scripts/miner_2_20340413_corr_regime_inverse_tail_skew_20v60obs_revalidation_signal.pkl'
j['validation']['period']='2020-01-01 through 2034-04-12; completed-session cutoff'
j['validation']['status']='EFFECTIVE'
j['validation']['regime_notes']='20-session admission remains valid on the full completed-session sample: 2026–28 IC/ICIR +0.05924/+0.15996 (252 dates), 2029–31 +0.02636/+0.07522 (283), and 2032–34 +0.01187/+0.03204 (194). Recent performance has weakened materially and should be monitored, but full-sample IC/ICIR remains above binding gates. No usable gated cross-sectional dates before 2026. Coverage 24.00%, with 9.16 names on eligible IC dates; correlation evidence complete against 28 other effective signals.'
s=j['benchmark_admission']['selected_metrics'];s.update({'ic':0.033873099922482636,'icir':0.09335781866484005,'metric_path':'validation.metrics.decay.20d','max_abs_library_correlation':0.4495968305164807,'correlation_path':'validation.metrics.max_abs_library_correlation','quality':0.003162158999364411})
open(p,'w').write(json.dumps(j,indent=2)+'\n')

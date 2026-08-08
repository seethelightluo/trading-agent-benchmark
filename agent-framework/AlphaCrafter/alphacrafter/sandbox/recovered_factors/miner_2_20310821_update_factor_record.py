import json
p='factors/miner_2_20310403_inverse_equity_stress_amplified_rate_transmission_residual_30.json'
with open(p,encoding='utf-8') as f: d=json.load(f)
m=d['validation']['metrics']
m.update({'ic':0.057478,'icir':0.195932,'ic_horizon_days':20,'ic_dates':878,'hit_ratio':0.5843,'mean_instruments':13.0,'turnover_mean_daily_rank':0.121512,'coverage':0.204662,'valid_factor_cells':11414,'max_abs_library_correlation':0.128667,'closest_library_factor':'dxy_shock_asymmetry','closest_common_valid_cells':10813})
m['decay']={'1_day':{'ic':0.019868,'icir':0.066564,'dates':878,'hit_ratio':0.5046},'5_day':{'ic':0.027184,'icir':0.085461,'dates':878,'hit_ratio':0.5467},'10_day':{'ic':0.038779,'icir':0.126956,'dates':878,'hit_ratio':0.5547},'20_day':{'ic':0.057478,'icir':0.195932,'dates':878,'hit_ratio':0.5843}}
d['version']='2031-08-21 revalidation'
d['last_validated']='2031-08-21T00:00:00Z'
d['revalidation_due']='2031-11-21'
d['validation']['period']='2026-07-16 to 2031-08-20 (point-in-time cutoff; 878 eligible forward-IC dates)'
d['validation']['regime_notes']='Revalidated 2031-08-21 with a point-in-time visibility cutoff of 2031-08-20. The reproducible run yielded the same 878 eligible IC observations as the prior validation, so reported estimates are unchanged apart from corrected output precision. Deployable inverse orientation passes at 20 days: IC +0.057478, ICIR +0.195932. 20-day 2026-2027: IC +0.062635, ICIR +0.184151, 366 dates; 2028-current: IC +0.053791, ICIR +0.210940, 512 dates. Complete reconstructed comparison with 25 admitted-library signals found maximum absolute Spearman rho 0.128667 against dxy_shock_asymmetry over 10,813 common valid cells (<0.5000). The 15-asset universe and selective rolling-beta coverage warrant conservative interpretation.'
d['benchmark_admission']['selected_metrics'].update({'ic':0.057478,'icir':0.195932,'max_abs_library_correlation':0.128667,'quality':0.057478*0.195932})
with open(p,'w',encoding='utf-8') as f: json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
print('updated',p,'status',d['validation']['status'],'quality',d['benchmark_admission']['selected_metrics']['quality'])

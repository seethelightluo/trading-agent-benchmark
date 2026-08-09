import json
from pathlib import Path
p=Path('factors/miner_2_20310403_inverse_equity_stress_amplified_rate_transmission_residual_30.json')
d=json.loads(p.read_text())
v=d['validation']; m=v['metrics']
d['version']='2032-01-22 revalidation'
d['last_validated']='2032-01-22T00:00:00Z'
d['revalidation_due']='2032-04-22'
v['period']='2026-07-16 to 2031-08-20 (latest available point-in-time completed-data endpoint; 878 eligible forward-IC dates; requested cutoff 2032-01-21)'
v['status']='EFFECTIVE'
v['regime_notes']='Revalidated 2032-01-22 using a requested point-in-time visibility cutoff of 2032-01-21. The visible completed-data endpoint remains 2031-08-20, hence this is an evidence-staleness check rather than an incremental sample update and statistics are unchanged. Deployable inverse orientation passes at 20 days: IC +0.057478, ICIR +0.195932. 20-day 2026-2027: IC +0.062635, ICIR +0.184151, 366 dates; 2028-data endpoint: IC +0.053791, ICIR +0.210940, 512 dates. Complete reconstructed comparison against 25 admitted-library signals found maximum absolute Spearman rho 0.128667 versus dxy_shock_asymmetry over 10,813 common valid cells (<0.5000). The intentional 15-asset universe averaged 13 valid instruments per IC date, and selective rolling-beta coverage warrants conservative interpretation.'
d['benchmark_admission']['selected_metrics'].update({'ic':m['ic'],'icir':m['icir'],'max_abs_library_correlation':m['max_abs_library_correlation'],'quality':abs(m['ic'])*abs(m['icir'])})
p.write_text(json.dumps(d,indent=2)+'\n')
print('updated',p,'status',v['status'],'quality',d['benchmark_admission']['selected_metrics']['quality'])

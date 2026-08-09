import json
from pathlib import Path
p=Path('factors/miner_2_20310403_inverse_equity_stress_amplified_rate_transmission_residual_30.json')
d=json.loads(p.read_text()); v=d['validation']; m=v['metrics']
d['version']='2032-02-19 revalidation'
d['last_validated']='2032-02-19T00:00:00Z'
d['revalidation_due']='2032-05-19'
v['period']='2026-07-16 to 2031-08-20 (latest available point-in-time completed-data endpoint; 878 eligible forward-IC dates; requested cutoff 2032-02-18)'
v['status']='EFFECTIVE'
v['regime_notes']='Revalidated 2032-02-19 using requested point-in-time completed-data cutoff 2032-02-18. Data service visible endpoint remains 2031-08-20, so this is an evidence-staleness check; all results match the 2032-01-22 run rather than representing incremental observations. The deployed inverse orientation passes: 20-day IC +0.057478 and ICIR +0.195932 (878 dates, 13.00 mean valid instruments). Decay: 1d +0.019868/+0.066564; 5d +0.027184/+0.085461; 10d +0.038779/+0.126956; 20d +0.057478/+0.195932 (IC/ICIR). 20-day regimes: 2026-27 +0.062635/+0.184151 over 366 dates; 2028 through endpoint +0.053791/+0.210940 over 512. Coverage 19.7748% (11,414 valid cells), selective due to rolling conditional-beta requirements; mean daily rank turnover 0.121512. A complete reconstructed screen of 25 admitted-library signal families finds maximum absolute Spearman correlation 0.128667 against dxy_shock_asymmetry across 10,813 common cells, below 0.5000. The sparse availability of newer source history means timeliness remains conditional on a future data refresh.'
d['benchmark_admission']['selected_metrics'].update({'ic':m['ic'],'icir':m['icir'],'max_abs_library_correlation':m['max_abs_library_correlation'],'quality':abs(m['ic'])*abs(m['icir'])})
p.write_text(json.dumps(d,indent=2)+'\n')
print('updated',d['version'],v['status'],d['last_validated'])

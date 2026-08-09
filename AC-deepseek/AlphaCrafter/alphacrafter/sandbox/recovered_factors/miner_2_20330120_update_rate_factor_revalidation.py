import json
from pathlib import Path
p=Path('factors/miner_2_20310403_inverse_equity_stress_amplified_rate_transmission_residual_30.json')
d=json.loads(p.read_text())
d['version']='2033-01-20 revalidation'
v=d['validation']; m=v['metrics']
v['period']='Point-in-time source history queried through 2033-01-19; latest available completed source session remained 2032-10-13 (878 eligible forward-IC dates; 15-instrument benchmark universe)'
v['status']='EFFECTIVE'
m.update({'ic':0.057478,'icir':0.195932,'ic_horizon_days':20,'ic_dates':878,'hit_ratio':0.5843,'mean_instruments':13.0,'turnover_mean_daily_rank':0.121512,'coverage':0.186138,'valid_factor_cells':11414,'max_abs_library_correlation':0.128667,'closest_library_factor':'dxy_shock_asymmetry','closest_common_valid_cells':10813})
v['regime_notes']='Scheduled fixed-specification revalidation executed 2033-01-20 with a point-in-time query through 2033-01-19. The locally available completed source history remained through 2032-10-13, so no new eligible forward-return observations were available: this is a timeliness check, not extrapolation. On 878 15-instrument-universe IC dates (mean 13.00 valid instruments, above the eight-instrument minimum), inverse orientation remains effective: 20d IC/ICIR +0.057478/+0.195932, 58.43% hit. Decay: 1d +0.019868/+0.066564; 5d +0.027184/+0.085461; 10d +0.038779/+0.126956; 20d +0.057478/+0.195932. At 20d, 2026-27 was +0.062635/+0.184151 (366 dates, 57.92% hit), and 2028-current was +0.053791/+0.210940 (512 dates, 58.79% hit). Full 25-family library screen has max absolute pooled Spearman correlation 0.128667 versus dxy_shock_asymmetry over 10,813 common cells, below 0.5000. Binding gates pass at 5d, 10d, and 20d; retained EFFECTIVE.'
d['last_validated']='2033-01-20T00:00:00Z';d['revalidation_due']='2033-04-20'
d['benchmark_admission']['selected_metrics'].update({'ic':0.057478,'icir':0.195932,'max_abs_library_correlation':0.128667,'quality':0.011261779496})
p.write_text(json.dumps(d,indent=2)+'\n')

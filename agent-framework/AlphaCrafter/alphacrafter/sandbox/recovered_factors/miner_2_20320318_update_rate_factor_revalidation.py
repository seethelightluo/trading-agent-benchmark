import json
from pathlib import Path
p=Path('factors/miner_2_20310403_inverse_equity_stress_amplified_rate_transmission_residual_30.json')
x=json.loads(p.read_text())
v=x['validation']; m=v['metrics']
x['version']='2032-03-18 revalidation'
v['period']='Point-in-time source history through 2032-03-17 (878 eligible forward-IC dates; 15-instrument benchmark universe)'
v['status']='EFFECTIVE'
m.update({'ic':0.057478,'icir':0.195932,'ic_horizon_days':20,'ic_dates':878,'hit_ratio':0.5843,'mean_instruments':13.0,'turnover_mean_daily_rank':0.121512,'coverage':0.196725,'valid_factor_cells':11414,'concentration':'Selective conditional-beta signal: 11,414 valid cells and 13.00 mean valid instruments on each of 878 IC dates, above the eight-instrument minimum. The lower panel coverage reflects availability requirements for rolling stress, rate-beta, and residual controls.','decay':{'1_day':{'ic':0.019868,'icir':0.066564,'dates':878,'hit_ratio':0.5046},'5_day':{'ic':0.027184,'icir':0.085461,'dates':878,'hit_ratio':0.5467},'10_day':{'ic':0.038779,'icir':0.126956,'dates':878,'hit_ratio':0.5547},'20_day':{'ic':0.057478,'icir':0.195932,'dates':878,'hit_ratio':0.5843}},'max_abs_library_correlation':0.128667,'closest_library_factor':'dxy_shock_asymmetry','closest_common_valid_cells':10813})
v['regime_notes']='Revalidated 2032-03-18 using completed daily data visible through 2032-03-17. The inverse orientation remains effective and passes the admission gates at 5, 10, and 20 sessions; selected 20-day IC/ICIR are +0.057478/+0.195932, with 58.43% positive daily ICs over 878 dates. Longer-horizon efficacy rises monotonically (1d +0.019868/+0.066564; 5d +0.027184/+0.085461; 10d +0.038779/+0.126956; 20d +0.057478/+0.195932). At 20 days, 2026-27: +0.062635/+0.184151, 366 dates, 57.92% hit; 2028-2032-03-17: +0.053791/+0.210940, 512 dates, 58.79% hit. A reconstructed 25-family library screen gives maximum absolute pooled Spearman correlation 0.128667 against dxy_shock_asymmetry (10,813 common cells), below 0.5000.'
x['last_validated']='2032-03-18T00:00:00Z';x['revalidation_due']='2032-06-18'
x['benchmark_admission']['selected_metrics'].update({'ic':0.057478,'icir':0.195932,'max_abs_library_correlation':0.128667,'quality':0.011261779496})
p.write_text(json.dumps(x,indent=2)+'\n')
print('updated',p)

import json
from pathlib import Path
p=Path('factors/miner_2_20310403_inverse_equity_stress_amplified_rate_transmission_residual_30.json')
d=json.loads(p.read_text())
d['version']='2033-07-07 revalidation'
d['last_validated']='2033-07-07T00:00:00Z'
d['revalidation_due']='2033-10-07'
v=d['validation']; m=v['metrics']
v['period']='Point-in-time source history queried through 2033-07-06; latest available completed source session remained 2032-10-13 (878 eligible forward-IC dates; 15-instrument benchmark universe)'
m['coverage']=0.180830
v['regime_notes']='Scheduled fixed-specification revalidation executed 2033-07-07 with a point-in-time query through 2033-07-06. The locally available completed source history remained through 2032-10-13, so no additional eligible forward-return observations were available; this is a timeliness check, not extrapolation. Across 878 IC dates in the 15-instrument benchmark universe (mean 13.00 valid instruments, above the eight-instrument minimum), inverse orientation remains effective at 20 sessions: IC/ICIR +0.057478/+0.195932, hit ratio 58.43%. Decay: 1d +0.019868/+0.066564; 5d +0.027184/+0.085461; 10d +0.038779/+0.126956; 20d +0.057478/+0.195932. At 20d, 2026-27 was +0.062635/+0.184151 (366 dates, 57.92% hit), and 2028-current was +0.053791/+0.210940 (512 dates, 58.79% hit). Full 25-family library screen maximum absolute pooled Spearman correlation was 0.128667 versus dxy_shock_asymmetry over 10,813 common cells, below 0.5000. Binding gates pass at 5d, 10d, and 20d; retained EFFECTIVE.'
p.write_text(json.dumps(d,indent=2)+'\n')
print('updated',p)

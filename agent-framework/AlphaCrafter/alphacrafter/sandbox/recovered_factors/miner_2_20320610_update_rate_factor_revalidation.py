import json
from pathlib import Path
p=Path('factors/miner_2_20310403_inverse_equity_stress_amplified_rate_transmission_residual_30.json')
d=json.loads(p.read_text())
d['version']='2032-06-10 revalidation'
d['last_validated']='2032-06-10T00:00:00Z'
d['revalidation_due']='2032-09-10'
d['validation']['period']='Point-in-time source history through 2032-03-17 (latest available completed source session; 878 eligible forward-IC dates; 15-instrument benchmark universe)'
d['validation']['regime_notes'] += ' Scheduled revalidation run 2032-06-10 found that the persistent source files still end at 2032-03-17; hence the point-in-time sample and all diagnostics are unchanged rather than using unavailable future observations. The factor remains EFFECTIVE under the binding gates.'
p.write_text(json.dumps(d,indent=2)+"\n")

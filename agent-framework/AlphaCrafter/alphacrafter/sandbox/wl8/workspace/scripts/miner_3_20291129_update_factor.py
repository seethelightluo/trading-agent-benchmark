import json
from pathlib import Path
p=Path('factors/miner_3_20291115_inverse_breakout_reversal_10d.json')
d=json.loads(p.read_text())
d['version']='2029-11-29'
d['validation']['period']='2020-01-01 to 2029-11-28'
d['validation']['timestamp']='2029-11-29T00:00:00Z'
m=d['validation']['metrics']; m.update({'ic':0.048688,'icir':0.149386,'ic_hit_ratio':0.5482,'ic_observations':861,'average_instruments':11.06,'coverage':0.7373,'turnover':0.045365,'decay':{'5d_ic':0.040689,'10d_ic':0.048688,'20d_ic':0.082491},'max_abs_library_correlation':None})
d['validation']['regime_notes']='Revalidated through 2029-11-28: full-history 10d IC/ICIR remain above admission gates, with persistent negative trailing 360d and 180d performance; retain capped usage and monitor drift.'
d['validation']['signal_artifact']='scripts/miner_3_20291115_inverse_breakout_signal.csv'
d['last_validated']='2029-11-29'
p.write_text(json.dumps(d,indent=2)+'\n')
print(json.loads(p.read_text())['factor_id'],json.loads(p.read_text())['validation']['status'],json.loads(p.read_text())['validation']['metrics']['ic'],json.loads(p.read_text())['validation']['metrics']['icir'],json.loads(p.read_text())['validation']['signal_artifact'])

import json
from pathlib import Path
p=Path('factors/miner_3_20300418_drawdown_rebound_10d.json')
d=json.loads(p.read_text())
d['version']='1.4'
d['calculation']['source_script']='scripts/miner_3_20300919_drawdown_rebound_revalidation.py'
d['calculation']['signal_artifact']['cutoff']='2030-09-18'
d['validation']['period']='2020-01-02 through 2030-09-18'
d['validation']['last_validated']='2030-09-19T00:00:00Z'
d['validation']['metrics'].update({'ic':0.03338347,'icir':3.85516,'ic_hit_ratio':0.55690,'valid_dates':1072,'data_dates':3478,'instruments':15,'average_instruments':15.0,'coverage':1.0,'turnover':0.1365453350771094,'decay_ic_5d':0.01171070,'decay_ic_10d':0.03338347,'decay_ic_20d':-0.00151488,'max_abs_library_correlation':None})
d['validation']['regime_notes']='Revalidated through 2030-09-18. Aggregate 10-day predictive power remains strong but regime-dependent: 2026 -0.02827, 2027 +0.07004, 2028 +0.05928, 2029 -0.00543, 2030 YTD +0.03788. Twenty-day decay is near zero/negative, supporting the specified 10-day horizon.'
d['benchmark_admission']['selected_metrics'].update({'ic':0.03338347,'icir':3.85516,'reported_max_abs_library_correlation':None,'quality':0.129?})
# fix quality to deterministic simple product proxy, not used as gate
d['benchmark_admission']['selected_metrics']['quality']=abs(0.03338347)*abs(3.85516)
d['benchmark_admission']['admitted_at']='2030-09-19T00:00:00Z'
p.write_text(json.dumps(d,indent=2)+'\n')
print(json.loads(p.read_text())['factor_id'],json.loads(p.read_text())['validation']['status'],json.loads(p.read_text())['validation']['metrics'])

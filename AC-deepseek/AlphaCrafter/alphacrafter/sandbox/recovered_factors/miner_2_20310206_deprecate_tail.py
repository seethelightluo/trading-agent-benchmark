import json
from pathlib import Path
p=Path('factors/miner_2_20300725_tail_correlation_asymmetry_residual_60.json')
d=json.loads(p.read_text())
d['version']='2031-02-06'
d['validation']['status']='DEPRECATED'
d['validation']['period']='2026-07-16 to 2031-02-05 (visible-data evaluation; historical forward returns only)'
d['validation']['metrics'].update({'ic':-0.095593,'icir':-0.324265,'ic_horizon_days':20,'ic_dates':1027,'hit_ratio':0.3739,'mean_instruments':11.59,'ic_standard_error':0.009199,'turnover_mean_daily_rank':0.106647,'coverage':0.222955,'valid_factor_cells':11966})
d['validation']['metrics']['decay']={'1_day':{'ic':-0.025133,'icir':-0.082386,'dates':1032,'hit_ratio':0.4516},'5_day':{'ic':-0.054881,'icir':-0.176837,'dates':1028,'hit_ratio':0.4212},'10_day':{'ic':-0.076464,'icir':-0.255369,'dates':1027,'hit_ratio':0.3982},'20_day':{'ic':-0.095593,'icir':-0.324265,'dates':1027,'hit_ratio':0.3739}}
d['validation']['regime_notes']='Revalidated 2031-02-06 through 2031-02-05. 20-day orientation remains negative and meets absolute admission magnitudes, but the 1-day ICIR is -0.082386, narrowly below the 0.0840 gate, and the factor is deprecated under the stated revalidation rule because ICIR is negative at all horizons. At 20 days: 2026-2027, 291 dates, IC -0.133480, ICIR -0.420477, hit 0.3230; 2028-2031-02-05, 736 dates, IC -0.080613, ICIR -0.283667, hit 0.3940. The original complete-library novelty evidence remains max_abs_library_correlation 0.122065; full correlation refresh exceeded the research runtime budget and no new admission is being considered.'
d['last_validated']='2031-02-06'
d['revalidation_due']=None
d['benchmark_admission']['selected_metrics']['ic']=-0.095593
d['benchmark_admission']['selected_metrics']['icir']=-0.324265
d['benchmark_admission']['selected_metrics']['quality']=abs(-0.095593)*abs(-0.324265)
out=Path('factors/miner_2_20300725_tail_correlation_asymmetry_residual_60_deprecated.json')
out.write_text(json.dumps(d,indent=2)+'\n')
p.unlink()
print(out)

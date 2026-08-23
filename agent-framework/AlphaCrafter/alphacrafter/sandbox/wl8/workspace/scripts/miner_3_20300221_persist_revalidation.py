import json
p='factors/miner_3_20300207_recovery_boost_10d.json'
with open(p) as f:d=json.load(f)
d['version']='20300221'
d['validation']['period']='2026-07-29 through 2030-02-06'
m=d['validation']['metrics'];m.update({'ic':0.053820,'icir':0.175990,'ic_hit_ratio':0.5475,'observations':906,'average_instruments':11.41,'coverage':0.7606,'turnover':0.056154,'signal_rows':15006})
d['validation']['regime_notes']='Revalidated 2030-02-21: positive 2026 IC 0.022741, 2027-28 0.074468, 2029 0.024183, trailing 360d 0.032656; trailing 180d mildly negative -0.005716. Predictive decay improved at 20d (IC 0.071759) versus 5d (0.035233); retain capped exposure.'
d['validation']['validation_timestamp']='2030-02-21T00:00:00Z'
d['validation']['signal_artifact']='scripts/miner_3_20300221_recovery_boost_signal.csv';d['validation']['ic_artifact']='scripts/miner_3_20300221_recovery_boost_ic.csv';d['validation']['provenance']='scripts/miner_3_20300221_revalidate_recovery.py'
d['benchmark_admission']['selected_metrics'].update({'ic':0.053820,'icir':0.175990,'reported_max_abs_library_correlation':None,'quality':0.009467})
with open(p,'w') as f:json.dump(d,f,indent=2)
with open(p) as f:x=json.load(f)
assert x['factor_id']=='miner_3_20300207_recovery_boost_10d' and x['validation']['status']=='EFFECTIVE' and x['validation']['metrics']['ic']>=.007 and x['validation']['metrics']['icir']>=.084 and x['validation']['signal_artifact']
print('verified',x['factor_id'],x['version'],x['validation']['status'],x['validation']['metrics']['ic'],x['validation']['metrics']['icir'],x['validation']['validation_timestamp'])

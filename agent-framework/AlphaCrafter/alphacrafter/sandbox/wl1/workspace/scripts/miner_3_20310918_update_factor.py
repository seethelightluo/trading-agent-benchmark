import json
f='factors/miner_3_20310904_recovery_pullback_20d.json'
with open(f,encoding='utf-8') as x: j=json.load(x)
v=j.setdefault('validation',{}); m=v.setdefault('metrics',{})
m.update({'ic':0.123152,'icir':0.364846,'horizon_days':20,'ic_observations':914,'unique_dates':914,'assets':15,'average_valid_instruments':9.4179,'coverage':0.245354,'turnover':0.071473,'hit_ratio':0.6171,'max_abs_library_correlation':None,'signal_artifact':'scripts/miner_3_20310918_recovery_pullback_signal.csv'})
v['status']='EFFECTIVE'; v['timestamp']='2031-09-18'; v['period']='2020-01-01 through 2031-09-17 available history; forward 20-day observations through validation sample'
v['validation_notes']='Revalidated on the 15-instrument cross-asset universe through 2031-09-17; dates with at least 8 valid instruments retained. Small cross-section implies conservative IC uncertainty. Deterministic library-correlation audit remains pending.'
v['regime_results']={'2026-2028':{'ic':0.063480,'icir':0.192390,'observations':477},'2029-2030':{'ic':0.181599,'icir':0.616999,'observations':321},'2031_ytd':{'ic':0.206788,'icir':0.485367,'observations':116}}
v['decay']={'1d_ic':0.004641,'5d_ic':0.041648,'10d_ic':0.052498,'20d_ic':0.123152}
j['last_validated']='2031-09-18'; j['version']='1.1'
with open(f,'w',encoding='utf-8') as x: json.dump(j,x,indent=2)
with open(f,encoding='utf-8') as x: q=json.load(x)
assert q['factor_id']=='miner_3_20310904_recovery_pullback_20d' and q['validation']['status']=='EFFECTIVE'
assert q['validation']['metrics']['ic']>=.007 and q['validation']['metrics']['icir']>=.084
assert q['validation']['metrics']['signal_artifact'].endswith('.csv')
print('verified',q['factor_id'],q['validation']['status'],q['validation']['metrics']['ic'],q['validation']['metrics']['icir'],q['validation']['metrics']['signal_artifact'],q['last_validated'])

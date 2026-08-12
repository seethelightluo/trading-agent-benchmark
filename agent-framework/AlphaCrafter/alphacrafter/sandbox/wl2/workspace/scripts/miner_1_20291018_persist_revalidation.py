import json
p='factors/miner_1_20291004_volume_amplified_residual_reversal_3d.json'
with open(p) as f: d=json.load(f)
m=d['validation']['metrics']
m['decay']='1-day IC 0.0670516186 (ICIR 0.1622422807); 5-day IC 0.1359896219 (ICIR 0.3273812405); both computed on 45 aligned dates, with 5-day endpoint available.'
m['decay_horizon_metrics']={'h1':{'ic':0.06705161862933628,'icir':0.16224228067274524},'h5':{'ic':0.13598962193776787,'icir':0.32738124049357653}}
m['revalidation_dates']=45
m['revalidation_average_instruments']=8.88888888888889
m['revalidation_coverage']=0.5925925925925927
m['revalidation_turnover_proxy']=1.1237933442298502
d['validation']['period']='2026-07-16 through 2029-10-18 (aligned observations; latest endpoint permitting h5 decay)'
d['validation']['regime_notes']='Revalidated 2029-10-18. Daily admission metrics remain above benchmark gates; five-day forward decay is stronger, but all usable observations remain concentrated in the simulator-aligned sample and coverage is 59.26%. Maintain conservative ensemble weight.'
d['last_validated']='2029-10-18T00:00:00Z'
d['signal_artifact']='scripts/miner_1_20291018_volume_residual3_signal.csv'
d['version']='1.1'
with open(p,'w') as f: json.dump(d,f,indent=2)
with open(p) as f: x=json.load(f)
assert x['factor_id']=='miner_1_20291004_volume_amplified_residual_reversal_3d'
assert x['validation']['status']=='EFFECTIVE'
assert x['validation']['metrics']['daily_paper_ic']>=.007 and x['validation']['metrics']['daily_paper_icir']>=.084
assert x['signal_artifact']
print('verified',x['factor_id'],x['validation']['status'],x['last_validated'],x['validation']['metrics']['daily_paper_ic'],x['validation']['metrics']['daily_paper_icir'])

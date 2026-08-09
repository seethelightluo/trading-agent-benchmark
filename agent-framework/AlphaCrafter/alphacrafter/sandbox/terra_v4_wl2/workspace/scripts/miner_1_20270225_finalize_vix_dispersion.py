import json, os, shutil
src='factors/miner_1_20270227_vix_dispersion_reversal10.json'; dst='factors/miner_1_20270225_vix_dispersion_reversal10.json'
with open(src) as f: d=json.load(f)
d['factor_id']='miner_1_20270225_vix_dispersion_reversal10'; d['last_validated']='2027-02-25T00:00:00Z'
art='../persistent/factor_signals_miner_1_20270225_vix_dispersion_reversal10.csv'
shutil.copyfile('../persistent/factor_signals_miner_3_20270226_vix_dispersion_reversal3.csv',art)
d['validation']['metrics']['signal_artifact']=art
d['validation']['signal_artifact']=art
with open(dst,'w') as f: json.dump(d,f,indent=2)
with open(dst) as f: z=json.load(f)
assert z['factor_id']=='miner_1_20270225_vix_dispersion_reversal10' and z['validation']['status']=='EFFECTIVE' and z['last_validated'].startswith('2027-02-25')
print('verified',z['factor_id'],z['validation']['metrics']['ic'],z['validation']['metrics']['icir'])

import json
d = json.load(open('factors/downbeta_spx_60.json'))
print('keys:', list(d.keys()))
print('validation keys:', list(d['validation'].keys()))
print('signal artifact present?', 'signal' in d or 'signal_path' in d or 'signal_artifact' in d)
print(json.dumps(d.get('validation',{}).get('metrics',{}), indent=1)[:1200])
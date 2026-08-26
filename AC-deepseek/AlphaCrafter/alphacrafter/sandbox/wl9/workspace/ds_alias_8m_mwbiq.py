import json
d = json.load(open('factors/vol_z_20d.json'))
print(json.dumps({k:v for k,v in d.items() if k!='signal_artifact'}, indent=1, default=str)[:4000])
print('...signal_artifact type:', type(d['validation'].get('signal_artifact')), 'len:', len(d['validation'].get('signal_artifact',[])) if isinstance(d['validation'].get('signal_artifact'), list) else 'n/a')
print('---example validation metrics---')
print(json.dumps(d['validation']['metrics'], indent=1, default=str)[:1500])
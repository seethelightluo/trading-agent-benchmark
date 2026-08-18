import json
d = json.load(open('factors/beta_ew_60d.json'))
print('keys:', list(d.keys()))
v = d.get('validation', {})
print('signal_artifact in validation:', 'signal_artifact' in v)
print('top-level signal_artifact:', 'signal_artifact' in d)
print(json.dumps(d.get('benchmark_admission', {}), indent=1)[:600])
print('metrics:', json.dumps(d.get('validation', {}).get('metrics'), indent=1)[:600])

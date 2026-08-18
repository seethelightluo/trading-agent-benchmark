import json
d = json.load(open('factors/rel_mom_20d_skip5.json'))
print('keys:', list(d.keys()))
print(json.dumps({k: (v if not isinstance(v, (dict, list)) else '...') for k, v in d.items()}, indent=1)[:800])
print('---validation keys:', list(d.get('validation', {}).keys()))
print(json.dumps(d.get('validation', {}), indent=1)[:1200])
print('---calculation:', json.dumps(d.get('calculation', {}), indent=1)[:400])
print('---parameters:', json.dumps(d.get('parameters', {}), indent=1)[:400])
print('---dependencies:', json.dumps(d.get('dependencies', {}), indent=1)[:400])
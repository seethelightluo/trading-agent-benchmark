import json
with open('factors/trend_r2_30_signed.json') as f:
    d = json.load(f)
print(json.dumps({k: d[k] for k in d if k != 'signal_artifacts'}, indent=1)[:3000])
print('---keys of validation---')
print(json.dumps(d['validation'], indent=1)[:1500])
print('---benchmark_admission---')
print(json.dumps(d.get('benchmark_admission'), indent=1)[:800])
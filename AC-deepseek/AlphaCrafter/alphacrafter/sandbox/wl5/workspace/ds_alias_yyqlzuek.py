import json
with open('factors/trend_r2_30_signed.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=1)[:3500])
print("KEYS:", list(d.keys()))
print("VALIDATION KEYS:", list(d.get('validation', {}).keys()))
print("METRICS:", json.dumps(d.get('validation', {}).get('metrics', {}), indent=1)[:1500])
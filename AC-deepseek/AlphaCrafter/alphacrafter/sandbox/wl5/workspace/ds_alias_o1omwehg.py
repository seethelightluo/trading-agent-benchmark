import json
with open('factors/trend_r2_30_signed.json') as f:
    d = json.load(f)
# Show structure keys
print(type(d))
if isinstance(d, dict):
    for k in d:
        v = d[k]
        if isinstance(v, dict):
            print(k, '->', list(v.keys())[:15])
        else:
            print(k, '->', str(v)[:200])
elif isinstance(d, list):
    print('list len', len(d))
    print(json.dumps(d[0], indent=1)[:1500])
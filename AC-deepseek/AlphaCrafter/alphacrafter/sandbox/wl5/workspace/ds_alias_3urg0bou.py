import json
d = json.load(open('../persistent/date.json'))
print(type(d), list(d.keys())[:10] if isinstance(d, dict) else d[:5])
if isinstance(d, dict):
    for k in d:
        v = d[k]
        print(k, type(v), (v[:3], v[-3:]) if isinstance(v, list) else v)

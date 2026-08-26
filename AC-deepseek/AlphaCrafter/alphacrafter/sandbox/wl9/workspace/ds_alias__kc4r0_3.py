import json
d = json.load(open('factors/vol_z_20d.json'))
print(type(d), list(d.keys()) if isinstance(d, dict) else len(d))
if isinstance(d, dict):
    for k,v in d.items():
        if isinstance(v, dict):
            print(k, '->', list(v.keys()))
        else:
            print(k, '->', str(v)[:200])
elif isinstance(d, list):
    print('list len', len(d), 'first elem keys:', list(d[0].keys()) if isinstance(d[0],dict) else type(d[0]))
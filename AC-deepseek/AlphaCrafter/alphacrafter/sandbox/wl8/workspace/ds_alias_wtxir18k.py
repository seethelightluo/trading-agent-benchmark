import json
for f in ['factors/flip_mom_20x10.json', 'factors/usdcny_beta_60.json']:
    try:
        d = json.load(open(f))
        print('=====', f)
        print('keys:', list(d.keys()))
        for k in d:
            v = d[k]
            if isinstance(v, dict):
                print(' ', k, '=> keys:', list(v.keys())[:20])
                for k2, v2 in list(v.items())[:8]:
                    print('     ', k2, ':', str(v2)[:150])
            else:
                print(' ', k, ':', str(v)[:200])
    except Exception as e:
        print(f, 'ERR', e)
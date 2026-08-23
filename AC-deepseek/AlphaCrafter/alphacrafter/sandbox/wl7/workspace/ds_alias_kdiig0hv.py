import json,glob
for f in sorted(glob.glob('factors/*.json')):
    if 'ensemble' in f: continue
    d=json.load(open(f))
    v=d.get('validation',{})
    print(d['factor_id'], '| dir=',d.get('expected_direction'), '| tags=',d.get('tags'))
    if isinstance(v,dict):
        for k in ('ic','icir','sharpe','direction','quality','hit_rate','coverage'):
            if k in v: print('    ',k,'=',v[k])
    elif isinstance(v,list):
        print('    validation is list len',len(v))
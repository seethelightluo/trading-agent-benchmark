import json, os, sys

for f in sorted(os.listdir('factors')):
    if f.endswith('.json') and not f.endswith('.bak') and f != 'factor_ensemble.json':
        p = os.path.join('factors', f)
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f, 'ERR', e)
            continue
        keys = list(d.keys())[:25]
        print('===', f, '| keys:', keys)
        meta = d.get('metadata') or d.get('meta') or {}
        if meta:
            print('   meta:', json.dumps(meta, default=str)[:1000])
        for k in ('ic','icir','quality','category','direction','window','sharpe','hit_rate','turnover','factor_id','name','description'):
            if k in d:
                v = d[k]
                print('   ', k, '=', str(v)[:300])
        for k in ('data','signal','values','history','series','exposures'):
            if k in d:
                v = d[k]
                if isinstance(v, dict):
                    print('   ', k, 'keys:', list(v.keys())[:10], 'n=', len(v))
                    # maybe last entries contain date/value
                    items = list(v.items())[-3:]
                    for ik, iv in items[:2]:
                        print('      sample', ik, '=', str(iv)[:120])
                elif isinstance(v, list):
                    print('   ', k, 'len=', len(v), 'sample:', str(v[-1])[:150] if v else None)
                break

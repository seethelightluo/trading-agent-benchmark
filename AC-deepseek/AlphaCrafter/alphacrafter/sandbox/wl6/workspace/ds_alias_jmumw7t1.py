import os, json
# Check factor files and their validation status + last_validated
for fn in sorted(os.listdir('factors')):
    if fn.endswith('.json') and not fn.endswith('.bak'):
        p = os.path.join('factors', fn)
        try:
            d = json.load(open(p))
            v = d.get('validation', {})
            m = v.get('metrics', {})
            print(fn, '| status:', v.get('status'), '| last_validated:', v.get('last_validated'), '| IC:', m.get('ic'), '| ICIR:', m.get('icir'))
        except Exception as e:
            print(fn, 'ERR', e)
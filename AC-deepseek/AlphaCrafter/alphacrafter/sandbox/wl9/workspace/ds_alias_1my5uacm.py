import json, glob, os
for f in sorted(glob.glob('factors/*.json')):
    if 'ensemble' in f or f.endswith('.bak'): continue
    try:
        dd = json.load(open(f))
        v = dd.get('validation', {})
        m = v.get('metrics', {})
        print(os.path.basename(f), '| status:', v.get('status'), '| lv:', dd.get('last_validated', v.get('validated_at','?')), '| IC:', m.get('ic'), '| ICIR:', m.get('icir'), '| rho:', m.get('max_abs_library_correlation'))
    except Exception as e:
        print(os.path.basename(f), 'ERR', str(e)[:80])
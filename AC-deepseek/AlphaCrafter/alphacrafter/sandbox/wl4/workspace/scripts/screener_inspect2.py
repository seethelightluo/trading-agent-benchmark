import json, os

for f in sorted(os.listdir('factors')):
    if f.endswith('.json') and not f.endswith('.bak') and f != 'factor_ensemble.json':
        p = os.path.join('factors', f)
        d = json.load(open(p))
        print('='*70)
        print('FACTOR:', f)
        print('  name:', d.get('factor_name'))
        print('  version:', d.get('version'))
        print('  expected_direction:', d.get('expected_direction'))
        print('  tags:', d.get('tags'))
        print('  parameters:', json.dumps(d.get('parameters'), default=str)[:400])
        val = d.get('validation')
        if val:
            print('  validation:', json.dumps(val, default=str)[:1500])
        adm = d.get('benchmark_admission')
        if adm:
            print('  benchmark_admission:', json.dumps(adm, default=str)[:600])

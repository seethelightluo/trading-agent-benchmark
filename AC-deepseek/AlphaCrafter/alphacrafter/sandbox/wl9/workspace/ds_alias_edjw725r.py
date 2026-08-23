import json, glob
for f in sorted(glob.glob('factors/*.json')):
    d = json.load(open(f))
    v = d.get('validation', {})
    if v.get('status') == 'EFFECTIVE' or f.endswith('factor_ensemble.json'):
        continue
    if v.get('status') != 'EFFECTIVE':
        continue
    calc = d.get('calculation', {})
    expr = calc.get('expression') if isinstance(calc, dict) else None
    print(d.get('factor_id'), '|', v.get('last_validated'), '|', v.get('metrics',{}).get('ic'), '|', v.get('metrics',{}).get('icir'), '|', str(expr)[:110])
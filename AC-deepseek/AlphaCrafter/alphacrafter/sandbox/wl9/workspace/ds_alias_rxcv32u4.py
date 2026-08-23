import json, glob
for f in sorted(glob.glob('factors/*.json')):
    if 'bak' in f or 'ensemble' in f or 'evicted' in f: continue
    d = json.load(open(f))
    calc = d.get('calculation', {})
    print(f"== {d.get('factor_id')}")
    print("   expr:", str(calc.get('expression', ''))[:200])
    print("   desc:", str(calc.get('description', ''))[:160])
    print("   params:", json.dumps(d.get('parameters', {}))[:200])
    deps = d.get('dependencies')
    if isinstance(deps, list): print("   deps:", deps)
    else: print("   deps:", str(deps)[:120])
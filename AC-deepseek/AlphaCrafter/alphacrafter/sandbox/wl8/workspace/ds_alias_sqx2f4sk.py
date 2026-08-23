import os, json, glob
# Look for any non-.bak JSON factor files anywhere in workspace
print("=== All .json files in factors/ (non-bak, non-dir) ===")
for fn in sorted(os.listdir('factors/')):
    p = os.path.join('factors', fn)
    if os.path.isfile(p) and fn.endswith('.json'):
        print(fn, os.path.getsize(p))

# Are there factor files elsewhere (persistent / scripts)?
print("\n=== Search for factor json files in scripts/ with factor metadata ===")
for p in sorted(glob.glob('scripts/*.json')):
    try:
        with open(p) as f:
            d = json.load(f)
        s = json.dumps(d)[:120].replace('\n',' ')
        print(os.path.basename(p), '->', s)
    except Exception as e:
        print(os.path.basename(p), 'ERR', e)
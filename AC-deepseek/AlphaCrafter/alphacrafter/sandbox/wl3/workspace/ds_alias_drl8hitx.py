import json, glob, os
for f in sorted(glob.glob('factors/*.json')):
    d = json.load(open(f))
    fid = d.get('factor_id','?')
    print("="*70)
    print(fid, "->", os.path.basename(f))
    for k,v in d.items():
        if k == 'signal_artifact': continue
        if isinstance(v, dict):
            print(f"  {k}: {json.dumps(v)[:400]}")
        else:
            print(f"  {k}: {str(v)[:200]}")

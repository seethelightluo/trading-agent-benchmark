import json, glob, os

files = sorted(glob.glob('factors/*.json'))
files = [f for f in files if not f.endswith('.bak') and '.npy' not in f]

# Dedupe by factor_id, prefer the newest (non-timestamped canonical or latest timestamp)
by_id = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    fid = d.get('factor_id')
    if fid is None:
        continue
    # prefer files without timestamp suffix (canonical)
    base = os.path.basename(f)
    has_ts = any(t in base for t in ['20260810', '20260811'])
    if fid not in by_id or (not has_ts and by_id[fid][1]):
        by_id[fid] = (d, has_ts)

print("UNIQUE FACTOR COUNT:", len(by_id))
print("=" * 100)
for fid, (d, _) in sorted(by_id.items()):
    v = d.get('validation', {})
    ba = d.get('benchmark_admission', {})
    tags = d.get('tags', {})
    params = d.get('parameters', {})
    print(f"\n--- {fid} ---")
    print("  name:", d.get('factor_name'))
    print("  category:", tags if isinstance(tags, str) else json.dumps(tags)[:150])
    print("  params:", json.dumps(params)[:200])
    print("  validation:", json.dumps(v)[:400])
    print("  admission:", json.dumps(ba)[:300])

import json, glob, os
# Check which factors in main dir have signal_artifact field (i.e., properly persisted)
for f in sorted(glob.glob('factors/*.json')):
    b = os.path.basename(f)
    if 'bak' in b or 'ensemble' in b: continue
    try:
        d = json.load(open(f))
        v = d.get('validation', {})
        has_art = 'signal_artifact' in v
        print(f"{b:45s} artifact={has_art} status={v.get('status')} last={d.get('last_validated')}")
    except Exception as e:
        print(f, 'ERR', e)

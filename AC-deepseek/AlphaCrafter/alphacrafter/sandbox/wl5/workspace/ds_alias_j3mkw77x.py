import json, glob, os
print("--- QUARANTINE REASONS ---")
for f in sorted(glob.glob('factors/quarantine/*.reason.json')):
    try:
        d = json.load(open(f))
        print(os.path.basename(f), '->', str(d)[:300])
    except Exception as e:
        print(f, 'ERR', e)

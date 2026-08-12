import glob, os, json
# list all current factor json files (non-bak, non-npy)
fids = []
for f in sorted(glob.glob('factors/*.json')):
    if '.bak' in f or 'reason' in f: continue
    try:
        d = json.load(open(f))
        if isinstance(d, dict) and 'factor_id' in d:
            fids.append(d['factor_id'])
    except Exception as e:
        pass
print('persisted factor ids:', sorted(set(fids)))

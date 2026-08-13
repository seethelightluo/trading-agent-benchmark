import json
# what does the audit say about recent cycle timing?
lines = open('factor_library_audit.jsonl').read().strip().split('\n')
for l in lines[-3:]:
    d = json.loads(l)
    print(json.dumps(d, indent=1)[:500])
print('---factor files---')
import glob, os
fs = sorted(glob.glob('factors/*.json'))
for f in fs:
    if 'bak' in f or 'signal' in f: continue
    print(os.path.basename(f))
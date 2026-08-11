import json, glob, os
for f in sorted(glob.glob('factors/evicted/*.reason.json')):
    d = json.load(open(f))
    print(os.path.basename(f).replace('.reason.json',''), '->', str(d)[:400])
    print()

import json, glob, os
# list all json factor files and their status
for p in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(p))
        status = d.get('validation',{}).get('status','?')
        fid = d.get('factor_id', os.path.basename(p))
        last = d.get('last_validated','?')
        print(f"{os.path.basename(p):45s} {status:12s} {last}")
    except Exception as e:
        print(f"{os.path.basename(p):45s} ERR {e}")

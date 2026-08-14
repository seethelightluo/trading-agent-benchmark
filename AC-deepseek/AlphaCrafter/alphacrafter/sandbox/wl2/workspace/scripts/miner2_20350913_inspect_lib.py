import json, glob, os

files = sorted(glob.glob('factors/*.json'))
print(f"Total json files: {len(files)}")
for f in files:
    try:
        d = json.load(open(f))
        v = d.get('validation', {})
        m = v.get('metrics', {})
        print(f"{os.path.basename(f):55s} id={d.get('factor_id','?'):30s} status={v.get('status','?'):12s} lv={d.get('last_validated', v.get('last_validated','?'))}")
    except Exception as e:
        print(f"{os.path.basename(f)}: ERROR {e}")

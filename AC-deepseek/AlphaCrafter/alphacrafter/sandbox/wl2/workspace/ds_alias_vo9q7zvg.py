import json, glob, os
# Check recent revalidation results
for f in sorted(glob.glob('scripts/miner_3_*revalidate*results*.json')):
    print('==', f, os.path.getmtime(f))
    try:
        with open(f) as fh:
            d = json.load(fh)
        if isinstance(d, dict):
            print(list(d.keys())[:20])
            for k in ('status_summary','summary','effective','deprecated'):
                if k in d:
                    print(k, ':', str(d[k])[:300])
        else:
            print(type(d), str(d)[:300])
    except Exception as e:
        print('ERR', e)
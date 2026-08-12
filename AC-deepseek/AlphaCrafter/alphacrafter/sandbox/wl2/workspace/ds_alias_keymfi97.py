import json, glob
for f in sorted(glob.glob('scripts/miner_3_202803*_results.json')):
    print('==', f)
    with open(f) as fh:
        d = json.load(fh)
    # print each factor status
    if isinstance(d, dict):
        for k,v in d.items():
            if isinstance(v, dict):
                print(' ', k, ':', v.get('status'), '| ic:', v.get('ic'), '| icir:', v.get('icir'), '| last_validated:', v.get('last_validated'))
            else:
                print(' ', k, ':', str(v)[:100])
    print()

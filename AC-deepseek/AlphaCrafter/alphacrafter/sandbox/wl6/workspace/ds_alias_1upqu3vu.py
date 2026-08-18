import json
for f in ['scripts/miner_2_20270408_screen_results.json']:
    d = json.load(open(f))
    print('==', f)
    print(json.dumps(d, indent=1)[:2500])
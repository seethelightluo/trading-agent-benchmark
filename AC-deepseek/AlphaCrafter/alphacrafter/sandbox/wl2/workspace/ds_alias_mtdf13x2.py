import json
for f in ['scripts/miner_3_20320122_screen_novel.json','scripts/miner_3_20320205_screen_novel2.json']:
    try:
        d = json.load(open(f))
        print("=== ", f)
        print(json.dumps(d, indent=1)[:1800])
    except Exception as e:
        print(f, "ERR", e)

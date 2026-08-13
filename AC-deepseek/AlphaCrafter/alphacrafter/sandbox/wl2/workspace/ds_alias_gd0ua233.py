import json
for f in ['scripts/miner_3_20320624_revalidate_results.json','scripts/miner_3_20320415_revalidate_results.json']:
    try:
        d = json.load(open(f))
        print("=== ", f)
        print(json.dumps(d, indent=1)[:2500])
    except Exception as e:
        print(f, "ERR", e)

import json
for f in ['scripts/miner2_cycle7_results.json','scripts/miner2_screen_all_v2_results.json']:
    try:
        d = json.load(open(f))
        print("="*15, f)
        print(json.dumps(d, indent=1)[:3000])
    except Exception as e:
        print(f, "ERR", e)
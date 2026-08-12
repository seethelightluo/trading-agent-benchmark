import os, json
# check miner_1 revalidate drift results
for p in ['scripts/miner_1_20280810_revalidate_drift.py', 'scripts/miner_1_20280323_revalidate_drift.json', 'scripts/miner_1_20280323_revalidate_drift.py']:
    if os.path.exists(p):
        print("="*20, p)
        if p.endswith('.json'):
            d = json.load(open(p))
            print(json.dumps(d, indent=1)[:1500])
        else:
            print(open(p).read()[:1500])
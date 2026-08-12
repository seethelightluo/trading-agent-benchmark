import json
for f in ['factors/miner2_20260715_nclv_1d.json','factors/miner2_20260715_rev_2d.json']:
    print("="*20, f)
    d = json.load(open(f))
    print(json.dumps(d, indent=1)[:2500])
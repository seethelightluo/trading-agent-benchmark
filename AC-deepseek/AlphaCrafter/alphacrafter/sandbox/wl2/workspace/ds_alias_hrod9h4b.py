import json
d=json.load(open('scripts/miner_3_20340803_revalidate_results.json'))
print(json.dumps(d, indent=1)[:3000])
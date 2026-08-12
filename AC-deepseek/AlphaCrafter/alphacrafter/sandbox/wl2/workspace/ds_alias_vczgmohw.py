import json
with open('scripts/miner_3_20290208_revalidate_results.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=1)[:2500])
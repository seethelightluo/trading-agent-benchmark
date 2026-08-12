import json
r = json.load(open('scripts/miner_3_20280810_results_batch2.json'))
print(json.dumps(r, indent=1)[:4000])
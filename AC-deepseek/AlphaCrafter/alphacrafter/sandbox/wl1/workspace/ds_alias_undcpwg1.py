import json
d = json.load(open('scripts/miner2_20280915_reval_results.json'))
print(json.dumps(d, indent=1)[:2500])
import json
with open('scripts/miner2_20291129_reval_results.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=2)[:4000])
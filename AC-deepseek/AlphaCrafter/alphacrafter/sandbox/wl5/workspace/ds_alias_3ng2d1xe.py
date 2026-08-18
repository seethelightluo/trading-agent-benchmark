import json
d = json.load(open('scripts/miner_1_20291018_revalidate_results.json'))
print(type(d))
if isinstance(d, dict):
    for k, v in list(d.items())[:15]:
        print(k, '->', str(v)[:300])
elif isinstance(d, list):
    for v in d[:15]:
        print(str(v)[:300])

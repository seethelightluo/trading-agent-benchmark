import json
d = json.load(open('scripts/miner3_20291109_screen1_results.json'))
print(type(d), list(d.keys())[:20] if isinstance(d, dict) else len(d))
if isinstance(d, dict):
    for k, v in list(d.items())[:15]:
        print(k, str(v)[:200])

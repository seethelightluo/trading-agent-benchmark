import json
d = json.load(open('scripts/miner_3_20290823_revalidate_results.json'))
print(type(d), list(d.keys()) if isinstance(d, dict) else len(d))
if isinstance(d, dict):
    for k, v in d.items():
        if isinstance(v, dict):
            print(k, '| ok:', v.get('ok'), '| IC:', round(v.get('ic',0),4), '| ICIR:', round(v.get('icir',0),4), '| n:', v.get('n_ic_dates'))
        else:
            print(k, v)

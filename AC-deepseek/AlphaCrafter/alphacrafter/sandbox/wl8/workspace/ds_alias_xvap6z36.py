import json
for p in ["factor_ensemble.json","factors/factor_ensemble.json"]:
    try:
        d=json.load(open(p))
        print(p, "->", json.dumps(d, indent=1)[:1500])
    except Exception as e:
        print(p, "ERR", e)
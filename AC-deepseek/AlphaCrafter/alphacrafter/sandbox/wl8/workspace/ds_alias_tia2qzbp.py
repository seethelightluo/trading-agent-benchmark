import json
for f in ["factors/flip_mom_20x10.json","factors/usdcny_beta_60.json"]:
    print("="*30, f)
    try:
        d = json.load(open(f))
        print(json.dumps(d, indent=1)[:2500])
    except Exception as e:
        print("ERR", e)
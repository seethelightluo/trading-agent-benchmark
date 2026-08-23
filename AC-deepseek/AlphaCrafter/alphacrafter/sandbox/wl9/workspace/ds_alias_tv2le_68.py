import json
with open("factors/bbz_20d.json.bak") as f:
    print(json.dumps(json.load(f), indent=2)[:2000])
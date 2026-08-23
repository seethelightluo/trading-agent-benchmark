import json
for f in ["factors/flip_mom_20x10.json", "factors/usdcny_beta_60.json"]:
    with open(f) as fh:
        d = json.load(fh)
    print(f, "->", d.get("factor_id"), d.get("validation", {}).get("status"), d.get("last_validated"))
    print(json.dumps(d.get("validation", {}), indent=1)[:1500])
    print("="*60)
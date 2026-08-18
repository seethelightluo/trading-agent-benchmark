import json
for fid in ["trend_r2_30_signed", "mom_10d_skip5", "tail_ratio_20"]:
    d = json.load(open(f"factors/{fid}.json"))
    print("="*80)
    print("factor_id:", d.get("factor_id"), "| status:", d.get("validation",{}).get("status"))
    print("keys:", list(d.keys()))
    vm = d.get("validation", {})
    print("validation keys:", list(vm.keys()))
    print(json.dumps(vm.get("metrics", {}), indent=1)[:1500])
    print("last_validated:", d.get("last_validated"))
    print("calculation:", json.dumps(d.get("calculation", {}))[:400])
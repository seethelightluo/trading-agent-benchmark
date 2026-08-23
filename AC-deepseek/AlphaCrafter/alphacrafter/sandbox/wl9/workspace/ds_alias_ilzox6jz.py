import json, os
factors = [f for f in os.listdir("factors/") if f.endswith(".json") and not f.endswith(".bak") and f != "factor_ensemble.json" and f != "evicted"]
for fn in sorted(factors):
    with open(f"factors/{fn}") as fh:
        d = json.load(fh)
        vid = d.get("validation",{})
        status = vid.get("status","?")
        ic = vid.get("metrics",{}).get("IC","?")
        icir = vid.get("metrics",{}).get("ICIR","?")
        lv = vid.get("last_validated","?")
        print(f"{fn:45s} status={status:12s} IC={str(ic):10s} ICIR={str(icir):10s} validated={lv}")
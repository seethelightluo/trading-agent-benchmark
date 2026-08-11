import json, glob
print("=== audit tail ===")
lines = open("factor_library_audit.jsonl").read().splitlines()
for line in lines[-4:]:
    print(line[:400])
print("=== quarantine ===")
for f in glob.glob("factors/quarantine/*"):
    print(f)
print("=== factor jsons ===")
for f in sorted(glob.glob("factors/*.json")):
    d = json.load(open(f))
    keys = list(d.keys())
    print(f, "| keys:", keys[:6], "| status:", d.get("validation", {}).get("status"), "| id:", d.get("factor_id"))

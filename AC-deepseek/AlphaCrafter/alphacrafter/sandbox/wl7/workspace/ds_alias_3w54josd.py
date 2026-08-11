import json, os, glob
print("=== audit tail ===")
for line in open("factor_library_audit.jsonl"):
    pass
print(line)
print("=== quarantine ===")
for f in glob.glob("factors/quarantine/*"):
    print(f)
print("=== factor jsons ===")
for f in sorted(glob.glob("factors/*.json")):
    d = json.load(open(f))
    print(f, d["validation"]["status"], d.get("factor_id"))

"""Probe current state: existing factors, data availability, current date."""
import os, json, glob
print("CWD:", os.getcwd())
print("FACTOR FILES:", glob.glob("factors/*.json"))
for f in sorted(glob.glob("factors/*.json")):
    try:
        d = json.load(open(f))
        print(f.split('/')[-1], "->", d.get("factor_id"), d.get("validation",{}).get("status"), "last:",
              d.get("last_validated"), "metrics:", d.get("validation",{}).get("metrics"))
    except Exception as e:
        print(f, "ERR", e)